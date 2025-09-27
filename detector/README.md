# AgentDebug Detector

A comprehensive error detection and analysis framework for debugging agent trajectories in interactive environments.

## Overview

The AgentDebug Detector provides a two-stage analysis pipeline for identifying and categorizing errors in agent execution traces. This framework helps developers understand failure patterns and improve agent performance across various interactive tasks.

## Components

### 1. Fine-Grained Analysis (`fine_grained_analysis.py`)
**Stage 1** - Performs detailed step-by-step analysis of agent trajectories to identify potential error patterns at each decision point.

Key features:
- Step-level error detection
- Module-specific analysis (Memory, Reflection, Planning, Action, System)
- Fine-grained error categorization
- Confidence scoring for detected issues

### 2. Critical Error Detection (`critical_error_detection.py`)
**Stage 2** - Identifies the critical failure point that led to task failure and provides root cause analysis.

Key features:
- Critical failure step identification
- Root cause analysis
- Error propagation tracking
- Failure module attribution

### 3. Error Definitions (`error_definitions.py`)
Comprehensive taxonomy of error types across different agent modules.

Error categories include:
- **Memory Errors**: Hallucination, memory retrieval failures, over-simplification
- **Reflection Errors**: Causal misattribution, outcome misinterpretation, progress misjudgment
- **Planning Errors**: Constraint ignorance, impossible actions, inefficient planning
- **Action Errors**: Format errors, invalid actions, misalignment, parameter errors
- **System Errors**: LLM limitations, environment errors, step limits, tool execution failures

## Usage

```python
from detector import fine_grained_analysis, critical_error_detection, error_definitions

# Load trajectory data
trajectory = load_agent_trajectory(...)

# Stage 1: Fine-grained analysis
step_errors = fine_grained_analysis.analyze(trajectory)

# Stage 2: Critical error detection
critical_failure = critical_error_detection.detect(trajectory, step_errors)

# Access error definitions
error_types = error_definitions.get_error_taxonomy()
```

## Supported Environments

- GAIA: General AI Assistant tasks
- AlfWorld: Embodied agent tasks
- WebShop: Web navigation and shopping tasks

## Performance Metrics

The detector achieves high accuracy in identifying critical failure points:
- Precision in critical step identification
- Comprehensive error coverage
- Module-specific accuracy metrics

## Citation

If you use this detector in your research, please cite:

```
@software{agentdebug2024,
  author = {m-serious},
  title = {AgentDebug: A Comprehensive Error Detection Framework},
  year = {2024},
  url = {https://github.com/ulab-uiuc/AgentDebug}
}
```

## License

This project is licensed under the MIT License.

## Contributing

Contributions are welcome! Please submit pull requests or open issues for bug reports and feature requests.

## Contact

For questions and support, please open an issue on the GitHub repository.
