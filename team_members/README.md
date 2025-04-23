# Team Members Workspace

This directory contains individual team member workspaces for exploration, experimentation, and development.

## Directory Structure

Each team member has their own directory:

- `miguel/` - Miguel's workspace
- `vetrivel/` - Vetrivel's workspace
- `lucy/` - Lucy's workspace
- `kaitlin/` - Kaitlin's workspace

## Recent Refactoring

The codebase has recently undergone a significant refactoring:

1. Unified the pipeline structure in `src/pipelines/`
2. Enhanced visualization tools in `src/utils/visualization.py`
3. Created a main entry point in `run_pipeline.py`
4. Moved reference implementations to team member directories

### Key Changes

- Baseline models are now properly organized in `src/models/baseline/`
- Feature extraction is now in dedicated modules in `src/features/`
- Visualization utilities are consolidated in `src/utils/visualization.py`
- Pipeline execution is standardized through `src/pipelines/run_baseline.py`

### Using Your Workspace

Team members should use their personal directories for experimentation and development. When your code is ready for integration into the main codebase, please follow the established structure for consistency. 