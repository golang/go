// compile -d=libfuzzer

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T struct {
	A
}

type A struct {
}

//go:noinline
func (a *A) Foo(s [2]string) {
}
