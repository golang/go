// run

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"strings"
)

type A interface{ A() }

type Impl struct{}

func (*Impl) A() {}

func main() {
	shouldNilPanic(func() {
		var v A
		v.A()
		v = &Impl{}
	})
	shouldNilPanic(func() {
		var v A
		defer func() {
			v = &Impl{}
		}()
		v.A()
	})
	shouldNilPanic(func() {
		var v A
		f := func() {
			v = &Impl{}
		}
		v.A()
		f()
	})
}

func shouldNilPanic(f func()) {
	defer func() {
		p := recover()
		if p == nil {
			panic("no nil deref panic")
		}
		if !strings.Contains(fmt.Sprintf("%s", p), "invalid memory address or nil pointer dereference") {
			panic(p)
		}
	}()
	f()
}
