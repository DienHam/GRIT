"""Experimental GRIT utilities for verl integration."""

from verl.experimental.grit.preservation import (
    PreservationBranchResult,
    TensorizedPreservationDataset,
    build_frozen_base_model,
    compute_preservation_branch_loss,
)
from verl.experimental.grit.predictor import (
    PredictorStepInfo,
    clone_current_gradients,
    combine_projected_task_and_preservation_gradients,
    temporary_predictor_step,
    write_final_grit_gradients,
)
from verl.experimental.grit.projector import (
    ProjectorAttachSummary,
    attach_projectors_to_mlp_linears,
    load_projectors,
    project_actor_mlp_gradients,
)

__all__ = [
    "PreservationBranchResult",
    "PredictorStepInfo",
    "ProjectorAttachSummary",
    "TensorizedPreservationDataset",
    "attach_projectors_to_mlp_linears",
    "build_frozen_base_model",
    "clone_current_gradients",
    "combine_projected_task_and_preservation_gradients",
    "compute_preservation_branch_loss",
    "load_projectors",
    "project_actor_mlp_gradients",
    "temporary_predictor_step",
    "write_final_grit_gradients",
]
