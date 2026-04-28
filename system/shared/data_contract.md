# Data Contract

The system accepts arbitrary CSV files and normalizes them into a target
database contract. Labels are not required during inference.

## Target Node Contract

Model-facing fields:

- `node_id`: stable node or transaction identifier.
- `timestamp`: optional event time or time bucket.
- `feature_*`: numeric model-facing attributes.

Display-only person fields:

- `display_name`
- `id_number`
- `phone`
- `region`
- `occupation`

Display-only fields are generated for public demo datasets when the source CSV
does not contain real person metadata. They are never used by feature
construction or inference.

## Target Edge Contract

Graph fields:

- `source_id`
- `target_id`
- `timestamp`
- `edge_type`
- `amount`

If an uploaded CSV has no explicit edge columns, the backend creates a small
preview graph from nearby rows for visualization only. The production inference
pipeline will use the dataset-specific graph construction code.
