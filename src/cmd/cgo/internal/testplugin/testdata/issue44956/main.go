// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Issue 44956: writable static temp is not exported correctly.
// In the test below, package base is
//
//     X = &map{...}
//
// which compiles to
//
//     X = &stmp           // static
//     stmp = makemap(...) // in init function
//
// plugin1 and plugin2 both import base. plugin1 doesn't use
// base.X, so that symbol is deadcoded in plugin1.
//
// plugin1 is loaded first. base.init runs at that point, which
// initialize base.stmp.
//
// plugin2 is then loaded. base.init already ran, so it doesn't run
// again. When base.stmp is not exported, plugin2's base.X points to
// its own private base.stmp, which is not initialized, fail.

package main

import "plugin"

func main() {
	_, err := plugin.Open("issue44956p1.so")
	if err != nil {
		panic("FAIL")
	}

	p2, err := plugin.Open("issue44956p2.so")
	if err != nil {
		panic("FAIL")
	}
	f, err := p2.Lookup("F")
	if err != nil {
		panic("FAIL")
	}
	x := f.(func() *map[int]int)()
	if x == nil || (*x)[123] != 456 {
		panic("FAIL")
	}
}
