// run

// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"reflect"
	"runtime"
)

func f(n int) int {
	return n % 2
}

func g(n int) int {
	return f(n)
}

func name(fn any) (res string) {
	return runtime.FuncForPC(uintptr(reflect.ValueOf(fn).Pointer())).Name()
}

func main() {
	println(name(f))
	println(name(g))
}
