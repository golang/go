// run

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The compiler was panicking on the wrong line number, where
// the panic was occurring in an inlined call.

package main

import (
	"runtime/debug"
	"strings"
)

type Wrapper struct {
	a []int
}

func (w Wrapper) Get(i int) int {
	return w.a[i]
}

func main() {
	defer func() {
		e := recover()
		if e == nil {
			panic("bounds check didn't fail")
		}
		stk := string(debug.Stack())
		if !strings.Contains(stk, "issue22083.go:40") {
			panic("wrong stack trace: " + stk)
		}
	}()
	foo := Wrapper{a: []int{0, 1, 2}}
	_ = foo.Get(0)
	_ = foo.Get(1)
	_ = foo.Get(2)
	_ = foo.Get(3) // stack trace should mention this line
}
