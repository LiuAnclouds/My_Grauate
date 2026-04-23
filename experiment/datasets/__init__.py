"""Dataset domain for the DyRIFT-GNN thesis benchmarks."""

from experiment.datasets.core.contracts import (
    PreparedGraphContract,
    PreparedPhaseContract,
    save_prepared_graph,
    save_prepared_phase,
    validate_prepared_graph,
    validate_prepared_phase,
)
from experiment.datasets.core.downloads import (
    DownloadResult,
    build_download_candidates,
    download_file,
    is_huggingface_url,
)
from experiment.datasets.core.registry import (
    DATASET_ENV_VAR,
    DatasetSpec,
    get_active_dataset_name,
    get_active_dataset_spec,
    get_dataset_spec,
    resolve_output_roots,
)

__all__ = [
    "DATASET_ENV_VAR",
    "DatasetSpec",
    "DownloadResult",
    "PreparedGraphContract",
    "PreparedPhaseContract",
    "build_download_candidates",
    "download_file",
    "get_active_dataset_name",
    "get_active_dataset_spec",
    "get_dataset_spec",
    "is_huggingface_url",
    "resolve_output_roots",
    "save_prepared_graph",
    "save_prepared_phase",
    "validate_prepared_graph",
    "validate_prepared_phase",
]
