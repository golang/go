These files are present to test building ssa on go files that use signatures from standard library packages.

Only the exported members used by the tests are needed.

Providing these decreases testing time ~10x (90s -> 8s) compared to building the standard library packages form source during tests.