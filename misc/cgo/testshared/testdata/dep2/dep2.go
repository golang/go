package dep2

import "testshared/depBase"

var W int = 1

var hasProg depBase.HasProg

type Dep2 struct {
	depBase.Dep
}

func G() int {
	return depBase.F() + 1
}
