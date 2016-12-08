package main

import (
	"depBase"
	"os"
	"reflect"
	"runtime"
)

// Having a function declared in the main package triggered
// golang.org/issue/18250
func DeclaredInMain() {
}

func main() {
	defer depBase.ImplementedInAsm()
	// This code below causes various go.itab.* symbols to be generated in
	// the executable. Similar code in ../depBase/dep.go results in
	// exercising https://github.com/golang/go/issues/17594
	reflect.TypeOf(os.Stdout).Elem()
	runtime.GC()
	depBase.V = depBase.F() + 1
}
