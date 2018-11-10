// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"reflect"

	"./p1"
)

func main() {
	var v1 = p1.S{1, 2}
	var v2 = struct { X, Y int }{1, 2}
	v1 = v2
	t1 := reflect.TypeOf(v1)
	t2 := reflect.TypeOf(v2)
	if !t1.AssignableTo(t2) {
		panic(0)
	}
	if !t2.AssignableTo(t1) {
		panic(1)
	}
}
