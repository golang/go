// run

// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build cgo

package main

import (
	"reflect"
	"runtime/cgo"
)

type E struct {
	x int
}

func (e *E) M() int {
	return e.x
}

type T struct {
	E
}

type NotInHeapT struct {
	E
	_ cgo.Incomplete
}

var (
	x = struct {
		E
		_ cgo.Incomplete
	}{E: E{x: 7}}
	tFn          = (*T).M
	notInHeapTFn = (*NotInHeapT).M
)

type I interface {
	M() int
}

//go:noinline
func call(i I) int {
	return i.M()
}

func main() {
	if got := call(&x); got != 7 {
		panic(got)
	}
	if reflect.ValueOf(tFn).Pointer() == reflect.ValueOf(notInHeapTFn).Pointer() {
		panic("pointer and scalar receiver wrappers were shared")
	}
}
