package ssagen

import (
	"cmd/compile/internal/ir"
	"cmd/compile/internal/ssa"
)

// goLocalAllocMap is mapping go_local variable to its need init ssa value.
var goLocalAllocMap = map[*ir.Name]*ssa.Value{}
