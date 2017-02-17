package depBase

import (
	"os"
	"reflect"
)

var SlicePtr interface{} = &[]int{}

var V int = 1

var HasMask []string = []string{"hi"}

type HasProg struct {
	array [1024]*byte
}

type Dep struct {
	X int
}

func (d *Dep) Method() int {
	// This code below causes various go.itab.* symbols to be generated in
	// the shared library. Similar code in ../exe/exe.go results in
	// exercising https://github.com/golang/go/issues/17594
	reflect.TypeOf(os.Stdout).Elem()
	return 10
}

func F() int {
	defer func() {}()
	return V
}
