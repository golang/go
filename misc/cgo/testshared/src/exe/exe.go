package main

import (
	"dep"
	"runtime"
)

func main() {
	defer dep.ImplementedInAsm()
	runtime.GC()
	dep.V = dep.F() + 1
}
