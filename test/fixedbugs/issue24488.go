// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"runtime"
	"strings"
)

type Func func()

func (f Func) Foo() {
	if f != nil {
		f()
	}
}

func (f Func) Bar() {
	if f != nil {
		f()
	}
	buf := make([]byte, 4000)
	n := runtime.Stack(buf, true)
	s := string(buf[:n])
	if strings.Contains(s, "-fm") {
		panic("wrapper present in stack trace:\n" + s)
	}
}

func main() {
	foo := Func(func() {})
	foo = foo.Bar
	foo.Foo()
}
