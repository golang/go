// errorcheck -0 -m -l

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis for function parameters.

package foo

var Ssink *string

type U struct {
	_sp  *string
	_spp **string
}

func A(sp *string, spp **string) U { // ERROR "leaking param: sp to result ~r2 level=0$" "leaking param: spp to result ~r2 level=0$"
	return U{sp, spp}
}

func B(spp **string) U { // ERROR "leaking param: spp to result ~r1 level=0$" "leaking param: spp to result ~r1 level=1$"
	return U{*spp, spp}
}

func tA1() {
	s := "cat"
	sp := &s   // ERROR "tA1 &s does not escape$"
	spp := &sp // ERROR "tA1 &sp does not escape$"
	u := A(sp, spp)
	_ = u
	println(s)
}

func tA2() {
	s := "cat"
	sp := &s   // ERROR "tA2 &s does not escape$"
	spp := &sp // ERROR "tA2 &sp does not escape$"
	u := A(sp, spp)
	println(*u._sp)
}

func tA3() {
	s := "cat"
	sp := &s   // ERROR "tA3 &s does not escape$"
	spp := &sp // ERROR "tA3 &sp does not escape$"
	u := A(sp, spp)
	println(**u._spp)
}

func tB1() {
	s := "cat"
	sp := &s   // ERROR "tB1 &s does not escape$"
	spp := &sp // ERROR "tB1 &sp does not escape$"
	u := B(spp)
	_ = u
	println(s)
}

func tB2() {
	s := "cat"
	sp := &s   // ERROR "tB2 &s does not escape$"
	spp := &sp // ERROR "tB2 &sp does not escape$"
	u := B(spp)
	println(*u._sp)
}

func tB3() {
	s := "cat"
	sp := &s   // ERROR "tB3 &s does not escape$"
	spp := &sp // ERROR "tB3 &sp does not escape$"
	u := B(spp)
	println(**u._spp)
}
