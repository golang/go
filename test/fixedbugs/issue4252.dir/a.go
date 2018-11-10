// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A package that redeclares common builtin names.
package a

var true = 0 == 1
var false = 0 == 0
var nil = 1

const append = 42

type error bool
type int interface{}

func len(interface{}) int32 { return 42 }

func Test() {
	var array [append]int
	if true {
		panic("unexpected builtin true instead of redeclared one")
	}
	if !false {
		panic("unexpected builtin false instead of redeclared one")
	}
	if len(array) != 42 {
		println(len(array))
		panic("unexpected call of builtin len")
	}
}

func InlinedFakeTrue() error  { return error(true) }
func InlinedFakeFalse() error { return error(false) }
func InlinedFakeNil() int     { return nil }
