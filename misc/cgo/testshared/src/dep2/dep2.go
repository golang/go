package dep2

import "dep"

var W int = 1

var hasProg dep.HasProg

func G() int {
	return dep.F() + 1
}
