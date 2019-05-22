// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

// // No C code required.
import "C"

import (
	"reflect"

	"testplugin/common"
)

func F() int {
	_ = make([]byte, 1<<21) // trigger stack unwind, Issue #18190.
	return 3
}

func ReadCommonX() int {
	return common.X
}

var Seven int

func call(fn func()) {
	fn()
}

func g() {
	common.X *= Seven
}

func init() {
	Seven = 7
	call(g)
}

type sameNameReusedInPlugins struct {
	X string
}

type sameNameHolder struct {
	F *sameNameReusedInPlugins
}

func UnexportedNameReuse() {
	h := sameNameHolder{}
	v := reflect.ValueOf(&h).Elem().Field(0)
	newval := reflect.New(v.Type().Elem())
	v.Set(newval)
}

func main() {
	panic("plugin1.main called")
}
