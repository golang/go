// compile -p=main

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type I interface {
	M() interface{}
}

type S1 struct{}

func (S1) M() interface{} {
	return nil
}

type EI interface{}

type S struct{}

func (S) M(as interface{ I }) {}

func f() interface{ EI } {
	return &S1{}
}

func main() {
	var i interface{ I }
	(&S{}).M(i)
}
