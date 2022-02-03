// run

// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

type S struct{
	m1Called, m2Called bool
}

func (s *S) M1(int) (int, int) {
	s.m1Called = true
	return 0, 0
}

func (s *S) M2(int) (int, int) {
	s.m2Called = true
	return 0, 0
}

type C struct {
	calls []func(int) (int, int)
}

func makeC() Funcs {
	return &C{}
}

func (c *C) Add(fn func(int) (int, int)) Funcs {
	c.calls = append(c.calls, fn)
	return c
}

func (c *C) Call() {
	for _, fn := range c.calls {
		fn(0)
	}
}

type Funcs interface {
	Add(func(int) (int, int)) Funcs
	Call()
}

func main() {
	s := &S{}
	c := makeC().Add(s.M1).Add(s.M2)
	c.Call()
	if !s.m1Called || !s.m2Called {
		panic("missed method call")
	}
}
