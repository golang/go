// errorcheckwithauto -0 -l -live -wb=0 -d=ssa/insert_resched_checks/off

//go:build !goexperiment.swissmap && !goexperiment.regabiargs

// For register ABI, liveness info changes slightly. See live_regabi.go.

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// non-swissmap-specific tests for live.go

package main

func printnl()

type T40 struct {
	m map[int]int
}

//go:noescape
func useT40(*T40)

func good40() {
	ret := T40{}              // ERROR "stack object ret T40$"
	ret.m = make(map[int]int) // ERROR "live at call to rand32: .autotmp_[0-9]+$" "stack object .autotmp_[0-9]+ runtime.hmap$"
	t := &ret
	printnl() // ERROR "live at call to printnl: ret$"
	// Note: ret is live at the printnl because the compiler moves &ret
	// from before the printnl to after.
	useT40(t)
}
