# Release checks

Checked on 2026-11-05:

- All Python sources parsed successfully.
- The Python wheel built successfully.
- All 15 methods and ablations completed the CPU synthetic smoke suite.

These checks cover execution and packaging, not real-data accuracy or CUDA behavior.
The release corrects the reference directory and module filenames from the supplied
folder. Training objectives and experiment configurations are retained.
