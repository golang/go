// errorcheckwithauto -0 -l -live -wb=0 -d=ssa/insert_resched_checks/off

//go:build goexperiment.swissmap && ((amd64 && goexperiment.regabiargs) || (arm64 && goexperiment.regabiargs))

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// swissmap-specific tests for live_regabi.go
// TODO(#54766): temporary while fast variants are disabled.

package main

func str() string

var b bool
var m2s map[string]*byte

func f17b(p *byte) { // ERROR "live at entry to f17b: p$"
	// key temporary
	if b {
		// TODO(go.dev/issue/54766): There is an extra autotmp here vs old maps.
		m2s[str()] = p // ERROR "live at call to mapassign: p$" "live at call to str: p$" "stack object .autotmp_1 string$" "stack object .autotmp_2 string$"

	}
	m2s[str()] = p // ERROR "live at call to mapassign: p$" "live at call to str: p$"
	m2s[str()] = p // ERROR "live at call to mapassign: p$" "live at call to str: p$"
}

func f17c() {
	// key and value temporaries
	if b {
		// TODO(go.dev/issue/54766): There is an extra autotmp here vs old maps.
		m2s[str()] = f17d() // ERROR "live at call to f17d: .autotmp_[0-9]+$" "live at call to mapassign: .autotmp_[0-9]+$" "stack object .autotmp_0 string$" "stack object .autotmp_1 string$"
	}
	m2s[str()] = f17d() // ERROR "live at call to f17d: .autotmp_[0-9]+$" "live at call to mapassign: .autotmp_[0-9]+$"
	m2s[str()] = f17d() // ERROR "live at call to f17d: .autotmp_[0-9]+$" "live at call to mapassign: .autotmp_[0-9]+$"
}

func f17d() *byte
