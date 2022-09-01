// run

// Copyright 2020 The Go Authors. All rights reserved.  Use of this
// source code is governed by a BSD-style license that can be found in
// the LICENSE file.

//go:build cgo && !aix

package main

import (
	"reflect"
	"runtime/cgo"
)

type NIH struct {
	_ cgo.Incomplete
}

var x, y NIH

func main() {
	if reflect.DeepEqual(&x, &y) != true {
		panic("should report true")
	}
}
