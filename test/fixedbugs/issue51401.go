// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 51401: bad inline info in generated interface method wrapper
// causes infinite loop in stack unwinding.

package main

import "runtime"

type Outer interface{ Inner }

type impl struct{}

func New() Outer { return &impl{} }

type Inner interface {
	DoStuff() error
}

func (a *impl) DoStuff() error {
	return newError()
}

func newError() error {
	stack := make([]uintptr, 50)
	runtime.Callers(2, stack[:])

	return nil
}

func main() {
	funcs := listFuncs(New())
	for _, f := range funcs {
		f()
	}
}

func listFuncs(outer Outer) []func() error {
	return []func() error{outer.DoStuff}
}
