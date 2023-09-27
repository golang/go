// run

// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type T1 struct{}
type T2 struct{}
type Both struct {
	T1
	T2
}

func (T1) m()   { panic("FAIL") }
func (T2) m()   { panic("FAIL") }
func (Both) m() {}

func f[T interface{ m() }](c T) {
	c.m()
}

func main() {
	var b Both
	b.m()
	f(b)
}
