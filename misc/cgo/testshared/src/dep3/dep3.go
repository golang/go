package dep3

// The point of this test file is that it references a type from
// depBase that is also referenced in dep2, but dep2 is loaded by the
// linker before depBase (because it is earlier in the import list).
// There was a bug in the linker where it would not correctly read out
// the type data in this case and later crash.

import (
	"dep2"
	"depBase"
)

type Dep3 struct {
	dep  depBase.Dep
	dep2 dep2.Dep2
}

func D3() int {
	var x Dep3
	return x.dep.X + x.dep2.X
}
