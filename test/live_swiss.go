// errorcheckwithauto -0 -l -live -wb=0 -d=ssa/insert_resched_checks/off

//go:build goexperiment.swissmap && !goexperiment.regabiargs

// For register ABI, liveness info changes slightly. See live_regabi.go.

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// swissmap-specific tests for live.go
// TODO(#54766): temporary while fast variants are disabled.

package main

// str is used to ensure that a temp is required for runtime calls below.
func str() string

var b bool
var m2 map[[2]string]*byte
var m2s map[string]*byte
var x2 [2]string

func f17b(p *byte) { // ERROR "live at entry to f17b: p$"
	// key temporary
	if b {
		// TODO(go.dev/issue/54766): There is an extra autotmp here vs old maps.
		m2s[str()] = p // ERROR "live at call to mapassign: p$" "live at call to str: p$" "stack object .autotmp_[0-9]+ string$"
	}
	m2s[str()] = p // ERROR "live at call to mapassign: p$" "live at call to str: p$"
	m2s[str()] = p // ERROR "live at call to mapassign: p$" "live at call to str: p$"
}

func f17c() {
	// key and value temporaries
	if b {
		// TODO(go.dev/issue/54766): There is an extra autotmp here vs old maps.
		m2s[str()] = f17d() // ERROR "live at call to f17d: .autotmp_[0-9]+$" "live at call to mapassign: .autotmp_[0-9]+$" "stack object .autotmp_[0-9]+ string$"
	}
	m2s[str()] = f17d() // ERROR "live at call to f17d: .autotmp_[0-9]+$" "live at call to mapassign: .autotmp_[0-9]+$"
	m2s[str()] = f17d() // ERROR "live at call to f17d: .autotmp_[0-9]+$" "live at call to mapassign: .autotmp_[0-9]+$"
}

func f17d() *byte

func printnl()

type T40 struct {
	m map[int]int
}

//go:noescape
func useT40(*T40)

func good40() {
	ret := T40{}              // ERROR "stack object ret T40$"
	ret.m = make(map[int]int) // ERROR "stack object .autotmp_[0-9]+ internal/runtime/maps.Map$"
	t := &ret
	printnl() // ERROR "live at call to printnl: ret$"
	// Note: ret is live at the printnl because the compiler moves &ret
	// from before the printnl to after.
	useT40(t)
}
