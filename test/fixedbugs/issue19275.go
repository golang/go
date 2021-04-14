// run

// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
)

type PI struct {
	Enabled bool
}

type SI struct {
	M map[string]*PI
}

//go:noinline
func (s *SI) test(name string) (*int, error) {
	n := new(int)
	*n = 99
	if err := addUpdate(n, s.M[name].Enabled, "enabled"); err != nil { // this was miscompiled
		return nil, fmt.Errorf(" error adding update for enable flag %t : %s",
			s.M[name].Enabled, err)
	}
	return n, nil
}

//go:noinline
func addUpdate(n *int, in interface{}, s ...string) error {
	if *n != 99 {
		println("FAIL, *n should be 99, not", *n)
	}
	return nil
}

func main1() {
	s := &SI{make(map[string]*PI)}
	s.M["dog"] = &PI{}
	s.test("dog")
}

//go:noinline
func g(b *byte, i interface{}) error {
	if *b != 17 {
		println("FAIL, *b should be 17, not", *b)
	}
	return nil
}

//go:noinline
func f(x *byte, m map[string]*bool) {
	if err := g(x, *m["hello"]); err != nil { // this was miscompiled
		return
	}
}

func main2() {
	m := make(map[string]*bool)
	x := false
	m["hello"] = &x
	b := byte(17)
	f(&b, m)
}

func main() {
	main2()
	main1()
}
