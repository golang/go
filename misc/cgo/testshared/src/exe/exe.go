package main

import (
	"depBase"
	"runtime"
)

func main() {
	defer depBase.ImplementedInAsm()
	runtime.GC()
	depBase.V = depBase.F() + 1
}
