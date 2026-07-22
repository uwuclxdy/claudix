use std::sync::Arc;

use arrow_array::types::Float32Type;
use arrow_array::{
    Array, ArrayRef, FixedSizeBinaryArray, FixedSizeListArray, Float32Array, RecordBatch,
    StringArray, UInt32Array, UInt64Array,
};
use arrow_schema::{DataType, Field, Schema, SchemaRef};
use futures::TryStreamExt;
use lancedb::Table;
use lancedb::query::{ExecutableQuery, QueryBase};

use crate::error::{ClaudixError, Result};
use crate::types::Dimension;

use super::chunk_row::{ChunkMetadata, StoredChunk, validate_vector};

pub(super) const FIELD_CHUNK_ID: &str = "chunk_id";
pub(super) const FIELD_FILE_PATH: &str = "file_path";
pub(super) const FIELD_LANGUAGE: &str = "language";
pub(super) const FIELD_KIND: &str = "kind";
pub(super) const FIELD_NAME: &str = "name";
pub(super) const FIELD_LINE_START: &str = "line_start";
pub(super) const FIELD_LINE_END: &str = "line_end";
pub(super) const FIELD_BYTE_START: &str = "byte_start";
pub(super) const FIELD_BYTE_END: &str = "byte_end";
pub(super) const FIELD_FILE_HASH: &str = "file_hash";
pub(super) const FIELD_CONTENT: &str = "content";
pub(super) const FIELD_VECTOR: &str = "vector";

pub(super) fn chunk_schema(dimension: Dimension) -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new(FIELD_CHUNK_ID, DataType::UInt64, false),
        Field::new(FIELD_FILE_PATH, DataType::Utf8, false),
        Field::new(FIELD_LANGUAGE, DataType::Utf8, false),
        Field::new(FIELD_KIND, DataType::Utf8, false),
        Field::new(FIELD_NAME, DataType::Utf8, true),
        Field::new(FIELD_LINE_START, DataType::UInt32, false),
        Field::new(FIELD_LINE_END, DataType::UInt32, false),
        Field::new(FIELD_BYTE_START, DataType::UInt32, false),
        Field::new(FIELD_BYTE_END, DataType::UInt32, false),
        Field::new(FIELD_FILE_HASH, DataType::FixedSizeBinary(16), false),
        Field::new(FIELD_CONTENT, DataType::Utf8, false),
        Field::new(
            FIELD_VECTOR,
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                i32::from(dimension.0),
            ),
            true,
        ),
    ]))
}

pub(super) fn record_batch_from_rows(
    rows: &[StoredChunk],
    dimension: Dimension,
) -> Result<RecordBatch> {
    for row in rows {
        validate_vector(&row.vector, dimension)?;
    }

    record_batch_from_rows_unchecked(rows, dimension)
}

pub(super) fn record_batch_from_rows_unchecked(
    rows: &[StoredChunk],
    dimension: Dimension,
) -> Result<RecordBatch> {
    let names: Vec<Option<String>> = rows.iter().map(|row| row.name.clone()).collect();
    let hash_refs: Vec<&[u8; 16]> = rows.iter().map(|row| &row.file_hash).collect();
    let vectors = rows
        .iter()
        .map(|row| Some(row.vector.iter().copied().map(Some).collect::<Vec<_>>()));

    RecordBatch::try_new(
        chunk_schema(dimension),
        vec![
            Arc::new(UInt64Array::from(
                rows.iter().map(|row| row.chunk_id).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| row.file_path.clone())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| row.language.clone())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(
                rows.iter().map(|row| row.kind.clone()).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(StringArray::from(names)) as ArrayRef,
            Arc::new(UInt32Array::from(
                rows.iter().map(|row| row.line_start).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt32Array::from(
                rows.iter().map(|row| row.line_end).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt32Array::from(
                rows.iter().map(|row| row.byte_start).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(UInt32Array::from(
                rows.iter().map(|row| row.byte_end).collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(
                FixedSizeBinaryArray::try_from_iter(hash_refs.into_iter())
                    .map_err(|error| ClaudixError::Store(error.to_string()))?,
            ) as ArrayRef,
            Arc::new(StringArray::from(
                rows.iter()
                    .map(|row| row.content.clone())
                    .collect::<Vec<_>>(),
            )) as ArrayRef,
            Arc::new(
                FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(
                    vectors,
                    i32::from(dimension.0),
                ),
            ) as ArrayRef,
        ],
    )
    .map_err(|error| ClaudixError::Store(error.to_string()))
}

pub(super) async fn read_all_rows(table: &Table) -> Result<Vec<StoredChunk>> {
    let batches = table
        .query()
        .limit(i64::MAX as usize)
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    batches_to_rows(batches)
}

/// Read every row satisfying `predicate`, an SQL filter pushed down to
/// LanceDB. Same output shape as [`read_all_rows`]; the caller owns building
/// (and escaping) the predicate.
pub(super) async fn read_rows_matching(table: &Table, predicate: &str) -> Result<Vec<StoredChunk>> {
    let batches = table
        .query()
        .only_if(predicate)
        .limit(i64::MAX as usize)
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await?;
    batches_to_rows(batches)
}

/// Projected read that fetches only scalar metadata columns, skipping the
/// embedding vector. Callers that only need `file_path`, `file_hash`,
/// `language`, and `name` should prefer this over [`read_all_rows`] to avoid
/// deserializing potentially large float arrays.
pub(super) async fn read_metadata_rows(table: &Table) -> Result<Vec<ChunkMetadata>> {
    let columns = [FIELD_FILE_PATH, FIELD_FILE_HASH, FIELD_LANGUAGE, FIELD_NAME]
        .map(str::to_owned)
        .to_vec();

    let batches = table
        .query()
        .select(lancedb::query::Select::Columns(columns))
        .limit(i64::MAX as usize)
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await?;

    batches_to_metadata_rows(batches)
}

fn batches_to_metadata_rows(batches: Vec<RecordBatch>) -> Result<Vec<ChunkMetadata>> {
    let mut rows = Vec::new();

    for batch in batches {
        for row_index in 0..batch.num_rows() {
            rows.push(ChunkMetadata {
                file_path: read_string(&batch, FIELD_FILE_PATH, row_index)?,
                file_hash: read_file_hash(&batch, row_index)?,
                language: read_string(&batch, FIELD_LANGUAGE, row_index)?,
                name: read_optional_string(&batch, FIELD_NAME, row_index)?,
            });
        }
    }

    Ok(rows)
}

/// Projected read that fetches every column except `content`.
///
/// The corpus-floor pass scores vectors and never touches the chunk text, which
/// is the column that makes a full read scale with the size of the source tree.
/// The omitted field comes back as an empty string.
pub(super) async fn read_rows_without_content(table: &Table) -> Result<Vec<StoredChunk>> {
    let columns = [
        FIELD_CHUNK_ID,
        FIELD_FILE_PATH,
        FIELD_LANGUAGE,
        FIELD_KIND,
        FIELD_NAME,
        FIELD_LINE_START,
        FIELD_LINE_END,
        FIELD_BYTE_START,
        FIELD_BYTE_END,
        FIELD_FILE_HASH,
        FIELD_VECTOR,
    ]
    .map(str::to_owned)
    .to_vec();

    let batches = table
        .query()
        .select(lancedb::query::Select::Columns(columns))
        .limit(i64::MAX as usize)
        .execute()
        .await?
        .try_collect::<Vec<_>>()
        .await?;

    rows_from_batches(batches, false)
}

pub(super) fn batches_to_rows(batches: Vec<RecordBatch>) -> Result<Vec<StoredChunk>> {
    rows_from_batches(batches, true)
}

fn rows_from_batches(batches: Vec<RecordBatch>, with_content: bool) -> Result<Vec<StoredChunk>> {
    let mut rows = Vec::new();

    for batch in batches {
        let dimension = vector_dimension(&batch)?;
        for row_index in 0..batch.num_rows() {
            let vector = read_vector(&batch, row_index)?;
            validate_vector(&vector, dimension)?;
            let content = if with_content {
                read_string(&batch, FIELD_CONTENT, row_index)?
            } else {
                String::new()
            };
            rows.push(StoredChunk {
                chunk_id: read_u64(&batch, FIELD_CHUNK_ID, row_index)?,
                file_path: read_string(&batch, FIELD_FILE_PATH, row_index)?,
                language: read_string(&batch, FIELD_LANGUAGE, row_index)?,
                kind: read_string(&batch, FIELD_KIND, row_index)?,
                name: read_optional_string(&batch, FIELD_NAME, row_index)?,
                line_start: read_u32(&batch, FIELD_LINE_START, row_index)?,
                line_end: read_u32(&batch, FIELD_LINE_END, row_index)?,
                byte_start: read_u32(&batch, FIELD_BYTE_START, row_index)?,
                byte_end: read_u32(&batch, FIELD_BYTE_END, row_index)?,
                file_hash: read_file_hash(&batch, row_index)?,
                content,
                vector,
            });
        }
    }

    Ok(rows)
}

fn read_string(batch: &RecordBatch, column: &str, row: usize) -> Result<String> {
    let array = batch
        .column_by_name(column)
        .ok_or_else(|| ClaudixError::Store(format!("missing column {column}")))?;
    let array = array
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| ClaudixError::Store(format!("column {column} was not utf8")))?;
    Ok(array.value(row).to_owned())
}

fn read_optional_string(batch: &RecordBatch, column: &str, row: usize) -> Result<Option<String>> {
    let array = batch
        .column_by_name(column)
        .ok_or_else(|| ClaudixError::Store(format!("missing column {column}")))?;
    let array = array
        .as_any()
        .downcast_ref::<StringArray>()
        .ok_or_else(|| ClaudixError::Store(format!("column {column} was not utf8")))?;

    if array.is_null(row) {
        return Ok(None);
    }

    Ok(Some(array.value(row).to_owned()))
}

fn read_u32(batch: &RecordBatch, column: &str, row: usize) -> Result<u32> {
    let array = batch
        .column_by_name(column)
        .ok_or_else(|| ClaudixError::Store(format!("missing column {column}")))?;
    let array = array
        .as_any()
        .downcast_ref::<UInt32Array>()
        .ok_or_else(|| ClaudixError::Store(format!("column {column} was not u32")))?;
    Ok(array.value(row))
}

fn read_u64(batch: &RecordBatch, column: &str, row: usize) -> Result<u64> {
    let array = batch
        .column_by_name(column)
        .ok_or_else(|| ClaudixError::Store(format!("missing column {column}")))?;
    let array = array
        .as_any()
        .downcast_ref::<UInt64Array>()
        .ok_or_else(|| ClaudixError::Store(format!("column {column} was not u64")))?;
    Ok(array.value(row))
}

fn read_file_hash(batch: &RecordBatch, row: usize) -> Result<[u8; 16]> {
    let array = batch
        .column_by_name(FIELD_FILE_HASH)
        .ok_or_else(|| ClaudixError::Store(format!("missing column {FIELD_FILE_HASH}")))?;
    let array = array
        .as_any()
        .downcast_ref::<FixedSizeBinaryArray>()
        .ok_or_else(|| {
            ClaudixError::Store("file_hash column was not fixed-size binary".to_owned())
        })?;

    <[u8; 16]>::try_from(array.value(row))
        .map_err(|_| ClaudixError::Store("file_hash value was not 16 bytes".to_owned()))
}

fn vector_dimension(batch: &RecordBatch) -> Result<Dimension> {
    let array = batch
        .column_by_name(FIELD_VECTOR)
        .ok_or_else(|| ClaudixError::Store(format!("missing column {FIELD_VECTOR}")))?;
    let array = array
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| ClaudixError::Store("vector column was not fixed-size list".to_owned()))?;
    u16::try_from(array.value_length())
        .map(Dimension)
        .map_err(|_| ClaudixError::Store("vector dimension overflowed u16".to_owned()))
}

fn read_vector(batch: &RecordBatch, row: usize) -> Result<Vec<f32>> {
    let array = batch
        .column_by_name(FIELD_VECTOR)
        .ok_or_else(|| ClaudixError::Store(format!("missing column {FIELD_VECTOR}")))?;
    let array = array
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .ok_or_else(|| ClaudixError::Store("vector column was not fixed-size list".to_owned()))?;
    if array.is_null(row) {
        return Err(ClaudixError::Store("vector value was null".to_owned()));
    }

    let values = array.value(row);
    let values = values
        .as_any()
        .downcast_ref::<Float32Array>()
        .ok_or_else(|| ClaudixError::Store("vector values were not float32".to_owned()))?;
    if values.null_count() > 0 {
        return Err(ClaudixError::Store(
            "vector values contained nulls".to_owned(),
        ));
    }

    Ok((0..values.len()).map(|index| values.value(index)).collect())
}
