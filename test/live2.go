// errorcheck -0 -live -wb=0

// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// liveness tests with inlining ENABLED
// see also live.go.

package main

// issue 8142: lost 'addrtaken' bit on inlined variables.

func printnl()

//go:noescape
func useT40(*T40)

type T40 struct {
	m map[int]int
}

func newT40() *T40 {
	ret := T40{}
	ret.m = make(map[int]int, 42) // ERROR "live at call to makemap: &ret$"
	return &ret
}

func bad40() {
	t := newT40() // ERROR "stack object ret T40$" "stack object .autotmp_[0-9]+ (runtime.hmap|internal/runtime/maps.Map)$"
	printnl()     // ERROR "live at call to printnl: ret$"
	useT40(t)
}

func good40() {
	ret := T40{}                  // ERROR "stack object ret T40$"
	ret.m = make(map[int]int, 42) // ERROR "stack object .autotmp_[0-9]+ (runtime.hmap|internal/runtime/maps.Map)$"
	t := &ret
	printnl() // ERROR "live at call to printnl: ret$"
	useT40(t)
}
