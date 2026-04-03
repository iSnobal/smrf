# Copilot Instructions for SMRF

## Repository Description
The Spatial Modeling for Resources Framework (SMRF) is a Python-based framework designed to provide a modular approach to spatial modeling of meteorological data. SMRF acts as the primary forcing data engine for the **Automated Water Supply Model (iSnobal/awsm)**, distributing point measurements or gridded data over a digital elevation model (DEM) to provide inputs for snow mass and energy balance models.

## Repository Structure
The project follows a modular Python package structure:
- `smrf/`: Main package directory.
  - `cli/`: Command-line interface implementations.
  - `data/`: Default configuration files, static resources, and sample data.
  - `distribute/`: Classes for distributing meteorological variables (temp, precip, etc.).
  - `envphys/`: Physical calculations for environmental variables (e.g., radiation, vapor pressure).
  - `framework/`: Core logic for model execution, threading, and data management.
  - `output/`: Handlers for various output formats (e.g., NetCDF).
  - `spatial/`: Spatial operations and handling of DEM/grid data.
  - `tests/`: Unit and integration tests.
  - `utils/`: Common utility functions and helpers.
- `docs/`: Sphinx documentation.
- `pyproject.toml` & `Makefile`: Build system and task automation.

## Key Guidelines

### 1. Code Style & Standards
Adhere to the specialized iSnobal organization agents defined in **`iSnobal/.github`**:
- **Python Style**: Follow `@iSnobal/.github/instructions/python-style-agent.md` for Ruff formatting, type hints, and naming conventions.
- **Legacy Migration**: Consult `@iSnobal/.github/instructions/legacy-migrator-agent.md` when refactoring complex or obscure legacy code.
- **Documentation**: Follow `@iSnobal/.github/instructions/documentation-agent.md` for NumPy-style docstrings and RST formatting.
- **Dependencies**: Consult `@iSnobal/.github/instructions/dependency-modernization-agent.md` for Conda-based environment management.
- **Performance**: Use `@iSnobal/.github/instructions/performance-cython-agent.md` for C/Cython optimizations and NumPy vectorization.
- **Snow Physics**: Defer to `@iSnobal/.github/instructions/snow-physics-agent.md` for physical correctness in radiation and energy balance logic.

### 2. Domain Context
- **Models**: Always consider the relationship between **SMRF** (forcing data/spatial modeling), `pysnobal` (iSnobal wrapper), and **iSnobal/awsm** (orchestrator). SMRF provides the critical spatial distribution of meteorological variables required by **iSnobal/awsm** to drive snow mass and energy balance models.
- **Topographic Context**: SMRF relies on **iSnobal/topocalc** for all topographic processing (slope, aspect, skyview, etc.). When performing domain context checks or modifying spatial calculations (e.g., in `smrf/data/load_topo.py` or `smrf/envphys/solar/toporad.py`), ensure the dependency on **iSnobal/topocalc** is maintained for physical consistency and topographic layer generation.
- **Config Files**: SMRF heavily relies on `.ini` configuration files (managed via `inicheck`). Ensure any changes to distribution methods or parameters are reflected in the expected `.ini` structure to maintain compatibility with model initialization and **iSnobal/awsm** orchestration workflows.

### 3. Review Style
When providing feedback or reviewing code:
- **Conciseness**: Be short and concise; explain the "why" behind recommendations.
- **Clarification**: Ask clarifying questions when code intent is unclear.
- **Efficiency**: Do not repeat comments that were previously resolved on new pushes.
- **Context**: Do not repeat any information that was already in the PR description.
- **Prioritization**: Focus on logic and physical correctness over purely technical changes.

### 4. Testing & Build
Follow the specialized `@iSnobal/.github/instructions/testing-coverage-agent.md` for detailed quality and coverage standards:
- **Framework**: Use the standard Python `unittest` framework for all tests.
- **Location**: Place new tests in `smrf/tests`.
- **Execution**:
  - Build extensions and run all tests: `make build_extensions tests`
  - Run specific test file: `python3 -m unittest smrf/tests/data/test_topo.py`
