// errorcheck -0 -live -wb=0

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// liveness tests with inlining ENABLED
// see also live.go.

package main

// issue 8142: lost 'addrtaken' bit on inlined variables.
// no inlining in this test, so just checking that non-inlined works.

func printnl()

type T40 struct {
	m map[int]int
}

func newT40() *T40 {
	ret := T40{}
	ret.m = make(map[int]int) // ERROR "live at call to makemap: &ret$"
	return &ret
}

func bad40() {
	t := newT40() // ERROR "live at call to makemap: .autotmp_[0-9]+ ret$"
	printnl()     // ERROR "live at call to printnl: .autotmp_[0-9]+ ret$"
	_ = t
}

func good40() {
	ret := T40{}
	ret.m = make(map[int]int) // ERROR "live at call to makemap: .autotmp_[0-9]+ ret$"
	t := &ret
	printnl() // ERROR "live at call to printnl: .autotmp_[0-9]+ ret$"
	_ = t
}
