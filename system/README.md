# DyRIFT Fraud Detection System

This folder contains the inference-stage system built around the trained
DyRIFT-TGAT full model artifacts in the parent experiment project.

The system stays under `experiment/system` so the research pipeline, trained
weights, generated metrics, and the demo application remain in one tracked
project without moving the existing experiment directories.

## Structure

- `backend/`: FastAPI service, SQLite database, upload pipeline, task status,
  and future model inference entry points.
- `frontend/`: React/Vite application for authentication, CSV upload, graph
  visualization, feature-processing progress, and inference result display.
- `shared/`: shared contracts for the target CSV/database schema.

## First Milestone

1. User registration and login endpoints with email verification-code flow.
2. CSV upload endpoint that stores raw rows, synthetic person metadata, graph
   nodes, graph edges, and model-facing feature JSON in SQLite.
3. Frontend workspace for upload, graph preview, processing progress, and
   inference result placeholders.

Full model loading and animated inference will be connected in later commits.
