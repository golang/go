// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"reflect"

	_ "./a"
	"./b"
)

var V struct{ i int }

func main() {
	if got := reflect.ValueOf(b.V).Type().Field(0).PkgPath; got != "b" {
		panic(`PkgPath=` + got + ` for first field of b.V, want "b"`)
	}
	if got := reflect.ValueOf(V).Type().Field(0).PkgPath; got != "main" {
		panic(`PkgPath=` + got + ` for first field of V, want "main"`)
	}
	if got := reflect.ValueOf(b.U).Type().Field(0).PkgPath; got != "b" {
		panic(`PkgPath=` + got + ` for first field of b.U, want "b"`)
	}
}
