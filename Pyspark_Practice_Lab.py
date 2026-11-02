"""
Standalone PySpark practice suite for core data engineering and replication patterns:
  1. CDC State Compaction & LSN Tie-Breaking (Out-of-Order & Duplicate Events)
  2. Dead-Letter Queue (DLQ) Routing & Schema Evolution Tolerance
  3. Rapid Payment Retry Detection (Window Functions: LEAD/LAG within 5 mins)
  4. Mitigating Data Skew in High-Volume Joins (Broadcast Join & Key Salting)
  5. Automated Source-vs-Target Financial Reconciliation (Row Checksums & Diffs)
  6. Streaming Micro-Batch Idempotent Upsert (foreachBatch MERGE simulation)
"""

import os
import sys
from tokenize import Double
from typing import Tuple

from pandas import isnull

# Ensure PySpark uses Homebrew OpenJDK 17 even if IDE terminal caches old JAVA_HOME
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@17/libexec/openjdk.jdk/Contents/Home"

try:
    from pyspark.sql import DataFrame, SparkSession, Window
    from pyspark.sql import functions as f
    from pyspark.sql.types import (
        DoubleType,
        LongType,
        MapType,
        StringType,
        StructField,
        StructType,
    )
except ImportError:
    print(
        "ERROR: PySpark is required to run this lab.\n"
        "Install it locally via: pip install pyspark\n"
        "Ensure Java 8/11/17 is installed and JAVA_HOME is set.",
        file=sys.stderr,
    )
    sys.exit(1)


def create_spark_session() -> SparkSession:
    """Initializes a Spark session for local fast testing."""
    spark = (
        SparkSession.builder.appName("PySpark_Practice_Lab")
        .master("local[*]")
        .config("spark.sql.shuffle.partitions", 4)
        .config("spark.sql.adaptive.enabled", "true")
        .config("spark.ui.showConsoleProgress", "false")
        .getOrCreate()
    )
    # Fix 1: sparkContext uses camelCase
    spark.sparkContext.setLogLevel("ERROR")
    print("Spark Session Intiated")
    return spark


def build_cdc_sample_data(spark: SparkSession) -> DataFrame:
    data = [
        # TXN_101: Out-of-order arrival + duplicate retry
        ("TXN_101", "M_01", 120.50, "CAPTURED", "u", "2026-10-30 10:00:05", 1002),
        ("TXN_101", "M_01", 120.50, "AUTHORIZED", "c", "2026-10-30 10:00:01", 1001),
        ("TXN_101", "M_01", 120.50, "CAPTURED", "u", "2026-10-30 10:00:05", 1002),
        # TXN_102: Tied timestamp! Status changed from AUTHORIZED -> SETTLED in same second
        ("TXN_102", "M_02", 450.00, "AUTHORIZED", "c", "2026-10-30 10:01:00", 2001),
        ("TXN_102", "M_02", 450.00, "SETTLED", "u", "2026-10-30 10:01:00", 2002),
        # TXN_103: Created, updated, and then hard-deleted ('d')
        ("TXN_103", "M_01", 75.00, "AUTHORIZED", "c", "2026-10-30 10:02:00", 3001),
        ("TXN_103", "M_01", 75.00, "VOIDED", "d", "2026-10-30 10:05:00", 3005),
    ]
    # Fix 2: Use StructField for transaction_id and merchant_id
    schema = StructType(
        [
            StructField("transaction_id", StringType(), False),
            StructField("merchant_id", StringType(), False),
            StructField("amount", DoubleType(), True),
            StructField("status", StringType(), False),
            StructField("op_type", StringType(), False),
            StructField("event_ts", StringType(), False),
            StructField("seq_num", LongType(), False),
        ]
    )
    return spark.createDataFrame(data, schema).withColumn(
        "event_ts", f.to_timestamp("event_ts")
    )


def solution_exercise_1_cdc_compaction(cdc_df: DataFrame) -> DataFrame:
    window_spec = Window.partitionBy("transaction_id").orderBy(
        f.col("event_ts").desc(), f.col("seq_num").desc()
    )
    return (
        cdc_df.withColumn("rn", f.row_number().over(window_spec))
        .filter(f.col("rn") == 1)
        .filter(f.col("op_type") != "d")
        .drop("rn")
        .orderBy("transaction_id")
    )
#*****************************************************************************
"""
Write the sample data and PySpark solution for Exercise 2: 
Dead-Letter Queue (DLQ) Routing & Schema Evolution Tolerance to add to my script.
"""
#********************************************************************************

def build_dlq_sample_data(spark: SparkSession) -> DataFrame:
    """Builds raw JSON ingestion events with schema drift and data quality errors."""
    raw_events = [
        # 1. Clean baseline record
        (1, '{"transaction_id": "TXN_201", "merchant_id": "M_01", "amount": 99.95, "currency": "USD"}'),
        # 2. Schema evolution: upstream added 'payment_method' and 'risk_score'
        (2, '{"transaction_id": "TXN_202", "merchant_id": "M_02", "amount": 250.00, "currency": "EUR", "payment_method": "APPLE_PAY", "risk_score": "0.12"}'),
        # 3. Malformed JSON syntax (missing closing brace)
        (3, '{"transaction_id": "TXN_203", "merchant_id": "M_01", "amount": 40.00'),
        # 4. Missing primary key (null transaction_id)
        (4, '{"transaction_id": null, "merchant_id": "M_03", "amount": 15.00, "currency": "USD"}'),
        # 5. Business rule violation (negative amount)
        (5, '{"transaction_id": "TXN_205", "merchant_id": "M_01", "amount": -50.00, "currency": "USD"}'),
        # 6. Type drift violation (non-numeric amount string)
        (6, '{"transaction_id": "TXN_206", "merchant_id": "M_04", "amount": "FREE", "currency": "USD"}'),
    ]

    schema = StructType(
            [
                StructField("ingestion_id", LongType(), False),
                StructField("raw_payload", StringType(), False)
            ]
    )
    return spark.createDataFrame(raw_events, schema)

def solution_exercise_2_dlq_routing(raw_df: DataFrame,) -> Tuple[DataFrame, DataFrame]:
    """
    Parses raw JSON payloads, preserves evolved schema fields in 'extra_fields',
    and splits records into (valid_df, dlq_df).
    """
    known_keys = ["transaction_id", "currency", "amount", "merchant_id"]

    # Parse JSON into a String->String map so malformed syntax yields NULL

    parsed_df = raw_df.withColumn(
        "raw_map",
        f.from_json(f.col("raw_payload"), MapType(StringType(), StringType())),
    )

    # Extract typed baseline columns and capture unannounced fields in extra_fields

    extracted_df = (
        parsed_df.withColumn("transaction_id", f.col("raw_map").getItem("transaction_id"))
        .withColumn("merchant_id", f.col("raw_map").getItem("merchant_id"))
        .withColumn("raw_amount", f.col("raw_map").getItem("amount"))
        .withColumn("amount", f.expr("try_cast(raw_amount as double)"))
        .withColumn("currency", f.col("raw_map").getItem("currency"))
        .withColumn(
            "extra_fields",
            f.map_filter(
                f.col("raw_map"),
                lambda k, v: ~k.isin(*known_keys),
            ),
        )
    )

    # Evaluate ordered data quality rules

    classified_df = extracted_df.withColumn(
        "dlq_reason",
        f.when(f.col("raw_map").isNull(), f.lit("Malformed_JSON"))
        .when(
            f.col("transaction_id").isNull() | (f.trim(f.col("transaction_id")) == ""),
            f.lit("Missing_Transaction_id")
        )
        .when(
            f.col("amount").isNull() | (f.col("amount") <= 0),
            f.lit("Invalid_Or_ Negative_Amount")
        )
        .otherwise(f.lit(None)),
    )

    valid_df = (
        classified_df.filter(f.col("dlq_reason").isNull())
        .select(
            "ingestion_id",
            "transaction_id",
            "merchant_id",
            "amount",
            "currency",
            "extra_fields"
        )
        .orderBy("ingestion_id")
    )

    dlq_df = (
        classified_df.filter(f.col("dlq_reason").isNotNull())
        .select("ingestion_id", "dlq_reason", "raw_payload")
        .orderBy("ingestion_id")
    )

    return valid_df, dlq_df


# ==============================================================================
# EXERCISE 3: Rapid Payment Retry Detection (Window LEAD/LAG)
# ==============================================================================
# Scenario (Classic PayPal Interview Question):
#   Write a PySpark transformation to identify payment attempts where a user
#   retried a payment within 5 minutes (<= 300 seconds) of a 'FAILED' attempt.
# ==============================================================================

def build_payments_attempt_data(spark: SparkSession) -> DataFrame:
    data = [
        ("U_1", "ATT_1", "FAILED", "2026-10-30 12:00:00"),
        ("U_1", "ATT_2", "FAILED", "2026-10-30 12:03:00"),   # Retried 180s after ATT_1 -> MATCH
        ("U_1", "ATT_3", "SUCCESS", "2026-10-30 12:04:30"),  # Retried 90s after ATT_2  -> MATCH
        ("U_2", "ATT_4", "FAILED", "2026-10-30 14:00:00"),
        ("U_2", "ATT_5", "SUCCESS", "2026-10-30 14:15:00"),  # Retried 900s later (> 300s) -> NO MATCH
        ("U_3", "ATT_6", "SUCCESS", "2026-10-30 15:00:00"),
        ("U_3", "ATT_7", "SUCCESS", "2026-10-30 15:01:00"),  # Previous was SUCCESS -> NO MATCH
    ] 
    return spark.createDataFrame(
        data, ["user_id", "attempt_id", "status", "attempt_ts"]
    ).withColumn("attempt_ts", f.to_timestamp("attempt_ts"))

def solution_exercise_3_retry(attempts_df: DataFrame) -> DataFrame:
    # Fix A: Use "user_id" (with underscore) to match the DataFrame schema
    user_window = Window.partitionBy("user_id").orderBy("attempt_ts")

    return (
        attempts_df.withColumn("next_attempt_id", f.lead("attempt_id").over(user_window))
        .withColumn("next_attempt_ts", f.lead("attempt_ts").over(user_window))
        .withColumn(
            # Fix B: Use "seconds_to_retry" (with underscore, not space)
            "seconds_to_retry",
            f.unix_timestamp("next_attempt_ts") - f.unix_timestamp("attempt_ts"),
        )
        .filter(
            (f.col("status") == "FAILED")
            & f.col("next_attempt_ts").isNotNull()
            & (f.col("seconds_to_retry") <= 300)
        )
        .select(
            "user_id",
            f.col("attempt_id").alias("failed_attempt_id"),
            "next_attempt_id",
            "seconds_to_retry",
        )
        .orderBy("user_id", "failed_attempt_id")
    )

if __name__ == "__main__":
    # Fix 3: Pass arguments between functions and display output
    spark_session = create_spark_session()
    sample_cdc_df = build_cdc_sample_data(spark_session)
    compacted_df = solution_exercise_1_cdc_compaction(sample_cdc_df)
    
    print("=== Exercise 1: Compacted CDC State ===")
    compacted_df.show(truncate=False)
    

    # exercise 2
    sample_dlq_df = build_dlq_sample_data(spark_session)
    valid_records_df, dlq_records_df = solution_exercise_2_dlq_routing(sample_dlq_df)
    print("=== Exercise 2A: Valid Evolved Records ===")
    valid_records_df.show(truncate=False)
    print("=== Exercise 2B: Dead-Letter Queue (DLQ)")
    dlq_records_df.show(truncate=False)

    # exercise 3
    sample_attempt_data = build_payments_attempt_data(spark_session)
    attempt_df = solution_exercise_3_retry(sample_attempt_data)
    print("=== Exercise 3: Rapid Payment Retry Detection")
    attempt_df.show(truncate=False)

    print("Closing Spark Session after execution")

    spark_session.stop()