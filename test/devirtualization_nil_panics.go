// run

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"runtime"
	"strings"
)

type A interface{ A() }

type Impl struct{}

func (*Impl) A() {}

type Impl2 struct{}

func (*Impl2) A() {}

func main() {
	shouldNilPanic(28, func() {
		var v A
		v.A()
		v = &Impl{}
	})
	shouldNilPanic(36, func() {
		var v A
		defer func() {
			v = &Impl{}
		}()
		v.A()
	})
	shouldNilPanic(43, func() {
		var v A
		f := func() {
			v = &Impl{}
		}
		v.A()
		f()
	})

	// Make sure that both devirtualized and non devirtualized
	// variants have the panic at the same line.
	shouldNilPanic(55, func() {
		var v A
		defer func() {
			v = &Impl{}
		}()
		v. // A() is on a sepearate line
			A()
	})
	shouldNilPanic(64, func() {
		var v A
		defer func() {
			v = &Impl{}
			v = &Impl2{} // assign different type, such that the call below does not get devirtualized
		}()
		v. // A() is on a sepearate line
			A()
	})
}

var cnt = 0

func shouldNilPanic(wantLine int, f func()) {
	cnt++
	defer func() {
		p := recover()
		if p == nil {
			panic("no nil deref panic")
		}
		if strings.Contains(fmt.Sprintf("%s", p), "invalid memory address or nil pointer dereference") {
			callers := make([]uintptr, 128)
			n := runtime.Callers(0, callers)
			callers = callers[:n]

			frames := runtime.CallersFrames(callers)
			line := -1
			for f, next := frames.Next(); next; f, next = frames.Next() {
				if f.Func.Name() == fmt.Sprintf("main.main.func%v", cnt) {
					line = f.Line
					break
				}
			}

			if line != wantLine {
				panic(fmt.Sprintf("invalid line number in panic = %v; want = %v", line, wantLine))
			}

			return
		}
		panic(p)
	}()
	f()
}
