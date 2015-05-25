package dep2

import "dep"

var W int = 1

func G() int {
	return dep.F() + 1
}
