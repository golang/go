// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package depBase

import (
	"os"
	"reflect"

	"testshared/depBaseInternal"
)

// Issue 61973: indirect dependencies are not initialized.
func init() {
	if !depBaseInternal.Initialized {
		panic("depBaseInternal not initialized")
	}
	if os.Stdout == nil {
		panic("os.Stdout is nil")
	}

	Initialized = true
}

var Initialized bool

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
	// exercising https://golang.org/issues/17594
	reflect.TypeOf(os.Stdout).Elem()
	return 10
}

func F() int {
	defer func() {}()
	return V
}
