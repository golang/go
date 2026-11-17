# Freeze independent release coverage before implementation

The frozen checker was written and executed against unchanged source
`e96a570da271c43a80e23bb1a188aac054acff25` before the new release workflow
or executor existed. It exited 1 and recorded the missing coverage in
`baseline-red.json`.

Its contract requires both drivers at O1/O2/O3 without short mode, the complete
unfiltered parity release run and clean-tip stamp, two independent reproducibility
hosts, and actual guarded LSP/QEMU/Wasm execution. It also keeps the stricter
M22 closure result separate from those independent jobs. Existing fixture
assertions and the 31 documented optimization-matrix exclusions stay unchanged.

This is static CI coverage evidence. It establishes neither a compiler failure
nor a hosted runtime pass. `frozen-checker.sha256` fixes the checker and initial
RED before the implementation commit. The checker takes one repository-root
argument and needs Python plus PyYAML; it runs no Go command or download.
