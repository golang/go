// errorcheck -0 -m -l

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis for struct function parameters.
// Note companion strict_param1 checks *struct function parameters with similar tests.

package notmain

var Ssink *string

type U struct {
	_sp  *string
	_spp **string
}

type V struct {
	_u   U
	_up  *U
	_upp **U
}

func (u U) SP() *string { // ERROR "leaking param: u to result ~r0 level=0$"
	return u._sp
}

func (u U) SPP() **string { // ERROR "leaking param: u to result ~r0 level=0$"
	return u._spp
}

func (u U) SPPi() *string { // ERROR "leaking param: u to result ~r0 level=1$"
	return *u._spp
}

func tSPPi() {
	s := "cat"        // ERROR "moved to heap: s$"
	ps := &s          // ERROR "&s escapes to heap$"
	pps := &ps        // ERROR "tSPPi &ps does not escape$"
	pu := &U{ps, pps} // ERROR "tSPPi &U literal does not escape$"
	Ssink = pu.SPPi()
}

func tiSPP() {
	s := "cat"        // ERROR "moved to heap: s$"
	ps := &s          // ERROR "&s escapes to heap$"
	pps := &ps        // ERROR "tiSPP &ps does not escape$"
	pu := &U{ps, pps} // ERROR "tiSPP &U literal does not escape$"
	Ssink = *pu.SPP()
}

// BAD: need fine-grained analysis to avoid spurious escape of ps
func tSP() {
	s := "cat"        // ERROR "moved to heap: s$"
	ps := &s          // ERROR "&s escapes to heap$" "moved to heap: ps$"
	pps := &ps        // ERROR "&ps escapes to heap$"
	pu := &U{ps, pps} // ERROR "tSP &U literal does not escape$"
	Ssink = pu.SP()
}

func (v V) u() U { // ERROR "leaking param: v to result ~r0 level=0$"
	return v._u
}

func (v V) UP() *U { // ERROR "leaking param: v to result ~r0 level=0$"
	return v._up
}

func (v V) UPP() **U { // ERROR "leaking param: v to result ~r0 level=0$"
	return v._upp
}

func (v V) UPPia() *U { // ERROR "leaking param: v to result ~r0 level=1$"
	return *v._upp
}

func (v V) UPPib() *U { // ERROR "leaking param: v to result ~r0 level=1$"
	return *v.UPP()
}

func (v V) USPa() *string { // ERROR "leaking param: v to result ~r0 level=0$"
	return v._u._sp
}

func (v V) USPb() *string { // ERROR "leaking param: v to result ~r0 level=0$"
	return v.u()._sp
}

func (v V) USPPia() *string { // ERROR "leaking param: v to result ~r0 level=1$"
	return *v._u._spp
}

func (v V) USPPib() *string { // ERROR "leaking param: v to result ~r0 level=1$"
	return v._u.SPPi()
}

func (v V) UPiSPa() *string { // ERROR "leaking param: v to result ~r0 level=1$"
	return v._up._sp
}

func (v V) UPiSPb() *string { // ERROR "leaking param: v to result ~r0 level=1$"
	return v._up.SP()
}

func (v V) UPiSPc() *string { // ERROR "leaking param: v to result ~r0 level=1$"
	return v.UP()._sp
}

func (v V) UPiSPd() *string { // ERROR "leaking param: v to result ~r0 level=1$"
	return v.UP().SP()
}

// BAD: need fine-grained (field-sensitive) analysis to avoid spurious escape of all but &s3
func tUPiSPa() {
	s1 := "ant"
	s2 := "bat"          // ERROR "moved to heap: s2$"
	s3 := "cat"          // ERROR "moved to heap: s3$"
	s4 := "dog"          // ERROR "moved to heap: s4$"
	s5 := "emu"          // ERROR "moved to heap: s5$"
	s6 := "fox"          // ERROR "moved to heap: s6$"
	ps2 := &s2           // ERROR "&s2 escapes to heap$"
	ps4 := &s4           // ERROR "&s4 escapes to heap$" "moved to heap: ps4$"
	ps6 := &s6           // ERROR "&s6 escapes to heap$" "moved to heap: ps6$"
	u1 := U{&s1, &ps2}   // ERROR "tUPiSPa &ps2 does not escape$" "tUPiSPa &s1 does not escape$"
	u2 := &U{&s3, &ps4}  // ERROR "&ps4 escapes to heap$" "&s3 escapes to heap$" "tUPiSPa &U literal does not escape$"
	u3 := &U{&s5, &ps6}  // ERROR "&U literal escapes to heap$" "&ps6 escapes to heap$" "&s5 escapes to heap$"
	v := &V{u1, u2, &u3} // ERROR "tUPiSPa &V literal does not escape$" "tUPiSPa &u3 does not escape$"
	Ssink = v.UPiSPa()   // Ssink = &s3 (only &s3 really escapes)
}

// BAD: need fine-grained (field-sensitive) analysis to avoid spurious escape of all but &s3
func tUPiSPb() {
	s1 := "ant"
	s2 := "bat"          // ERROR "moved to heap: s2$"
	s3 := "cat"          // ERROR "moved to heap: s3$"
	s4 := "dog"          // ERROR "moved to heap: s4$"
	s5 := "emu"          // ERROR "moved to heap: s5$"
	s6 := "fox"          // ERROR "moved to heap: s6$"
	ps2 := &s2           // ERROR "&s2 escapes to heap$"
	ps4 := &s4           // ERROR "&s4 escapes to heap$" "moved to heap: ps4$"
	ps6 := &s6           // ERROR "&s6 escapes to heap$" "moved to heap: ps6$"
	u1 := U{&s1, &ps2}   // ERROR "tUPiSPb &ps2 does not escape$" "tUPiSPb &s1 does not escape$"
	u2 := &U{&s3, &ps4}  // ERROR "&ps4 escapes to heap$" "&s3 escapes to heap$" "tUPiSPb &U literal does not escape$"
	u3 := &U{&s5, &ps6}  // ERROR "&U literal escapes to heap$" "&ps6 escapes to heap$" "&s5 escapes to heap$"
	v := &V{u1, u2, &u3} // ERROR "tUPiSPb &V literal does not escape$" "tUPiSPb &u3 does not escape$"
	Ssink = v.UPiSPb()   // Ssink = &s3 (only &s3 really escapes)
}

// BAD: need fine-grained (field-sensitive) analysis to avoid spurious escape of all but &s3
func tUPiSPc() {
	s1 := "ant"
	s2 := "bat"          // ERROR "moved to heap: s2$"
	s3 := "cat"          // ERROR "moved to heap: s3$"
	s4 := "dog"          // ERROR "moved to heap: s4$"
	s5 := "emu"          // ERROR "moved to heap: s5$"
	s6 := "fox"          // ERROR "moved to heap: s6$"
	ps2 := &s2           // ERROR "&s2 escapes to heap$"
	ps4 := &s4           // ERROR "&s4 escapes to heap$" "moved to heap: ps4$"
	ps6 := &s6           // ERROR "&s6 escapes to heap$" "moved to heap: ps6$"
	u1 := U{&s1, &ps2}   // ERROR "tUPiSPc &ps2 does not escape$" "tUPiSPc &s1 does not escape$"
	u2 := &U{&s3, &ps4}  // ERROR "&ps4 escapes to heap$" "&s3 escapes to heap$" "tUPiSPc &U literal does not escape$"
	u3 := &U{&s5, &ps6}  // ERROR "&U literal escapes to heap$" "&ps6 escapes to heap$" "&s5 escapes to heap$"
	v := &V{u1, u2, &u3} // ERROR "tUPiSPc &V literal does not escape$" "tUPiSPc &u3 does not escape$"
	Ssink = v.UPiSPc()   // Ssink = &s3 (only &s3 really escapes)
}

// BAD: need fine-grained (field-sensitive) analysis to avoid spurious escape of all but &s3
func tUPiSPd() {
	s1 := "ant"
	s2 := "bat"          // ERROR "moved to heap: s2$"
	s3 := "cat"          // ERROR "moved to heap: s3$"
	s4 := "dog"          // ERROR "moved to heap: s4$"
	s5 := "emu"          // ERROR "moved to heap: s5$"
	s6 := "fox"          // ERROR "moved to heap: s6$"
	ps2 := &s2           // ERROR "&s2 escapes to heap$"
	ps4 := &s4           // ERROR "&s4 escapes to heap$" "moved to heap: ps4$"
	ps6 := &s6           // ERROR "&s6 escapes to heap$" "moved to heap: ps6$"
	u1 := U{&s1, &ps2}   // ERROR "tUPiSPd &ps2 does not escape$" "tUPiSPd &s1 does not escape$"
	u2 := &U{&s3, &ps4}  // ERROR "&ps4 escapes to heap$" "&s3 escapes to heap$" "tUPiSPd &U literal does not escape$"
	u3 := &U{&s5, &ps6}  // ERROR "&U literal escapes to heap$" "&ps6 escapes to heap$" "&s5 escapes to heap$"
	v := &V{u1, u2, &u3} // ERROR "tUPiSPd &V literal does not escape$" "tUPiSPd &u3 does not escape$"
	Ssink = v.UPiSPd()   // Ssink = &s3 (only &s3 really escapes)
}

func (v V) UPiSPPia() *string { // ERROR "leaking param: v to result ~r0 level=2$"
	return *v._up._spp
}

func (v V) UPiSPPib() *string { // ERROR "leaking param: v to result ~r0 level=2$"
	return v._up.SPPi()
}

func (v V) UPiSPPic() *string { // ERROR "leaking param: v to result ~r0 level=2$"
	return *v.UP()._spp
}

func (v V) UPiSPPid() *string { // ERROR "leaking param: v to result ~r0 level=2$"
	return v.UP().SPPi()
}

// BAD: need fine-grained (field-sensitive) analysis to avoid spurious escape of all but &s4
func tUPiSPPia() {
	s1 := "ant"
	s2 := "bat"
	s3 := "cat"
	s4 := "dog"          // ERROR "moved to heap: s4$"
	s5 := "emu"          // ERROR "moved to heap: s5$"
	s6 := "fox"          // ERROR "moved to heap: s6$"
	ps2 := &s2           // ERROR "tUPiSPPia &s2 does not escape$"
	ps4 := &s4           // ERROR "&s4 escapes to heap$"
	ps6 := &s6           // ERROR "&s6 escapes to heap$" "moved to heap: ps6$"
	u1 := U{&s1, &ps2}   // ERROR "tUPiSPPia &ps2 does not escape$" "tUPiSPPia &s1 does not escape$"
	u2 := &U{&s3, &ps4}  // ERROR "tUPiSPPia &U literal does not escape$" "tUPiSPPia &ps4 does not escape$" "tUPiSPPia &s3 does not escape$"
	u3 := &U{&s5, &ps6}  // ERROR "&ps6 escapes to heap$" "&s5 escapes to heap$" "tUPiSPPia &U literal does not escape$"
	v := &V{u1, u2, &u3} // ERROR "tUPiSPPia &V literal does not escape$" "tUPiSPPia &u3 does not escape$"
	Ssink = v.UPiSPPia() // Ssink = *&ps4 = &s4 (only &s4 really escapes)
}

// BAD: need fine-grained (field-sensitive) analysis to avoid spurious escape of all but &s4
func tUPiSPPib() {
	s1 := "ant"
	s2 := "bat"
	s3 := "cat"
	s4 := "dog"          // ERROR "moved to heap: s4$"
	s5 := "emu"          // ERROR "moved to heap: s5$"
	s6 := "fox"          // ERROR "moved to heap: s6$"
	ps2 := &s2           // ERROR "tUPiSPPib &s2 does not escape$"
	ps4 := &s4           // ERROR "&s4 escapes to heap$"
	ps6 := &s6           // ERROR "&s6 escapes to heap$" "moved to heap: ps6$"
	u1 := U{&s1, &ps2}   // ERROR "tUPiSPPib &ps2 does not escape$" "tUPiSPPib &s1 does not escape$"
	u2 := &U{&s3, &ps4}  // ERROR "tUPiSPPib &U literal does not escape$" "tUPiSPPib &ps4 does not escape$" "tUPiSPPib &s3 does not escape$"
	u3 := &U{&s5, &ps6}  // ERROR "&ps6 escapes to heap$" "&s5 escapes to heap$" "tUPiSPPib &U literal does not escape$"
	v := &V{u1, u2, &u3} // ERROR "tUPiSPPib &V literal does not escape$" "tUPiSPPib &u3 does not escape$"
	Ssink = v.UPiSPPib() // Ssink = *&ps4 = &s4 (only &s4 really escapes)
}

// BAD: need fine-grained (field-sensitive) analysis to avoid spurious escape of all but &s4
func tUPiSPPic() {
	s1 := "ant"
	s2 := "bat"
	s3 := "cat"
	s4 := "dog"          // ERROR "moved to heap: s4$"
	s5 := "emu"          // ERROR "moved to heap: s5$"
	s6 := "fox"          // ERROR "moved to heap: s6$"
	ps2 := &s2           // ERROR "tUPiSPPic &s2 does not escape$"
	ps4 := &s4           // ERROR "&s4 escapes to heap$"
	ps6 := &s6           // ERROR "&s6 escapes to heap$" "moved to heap: ps6$"
	u1 := U{&s1, &ps2}   // ERROR "tUPiSPPic &ps2 does not escape$" "tUPiSPPic &s1 does not escape$"
	u2 := &U{&s3, &ps4}  // ERROR "tUPiSPPic &U literal does not escape$" "tUPiSPPic &ps4 does not escape$" "tUPiSPPic &s3 does not escape$"
	u3 := &U{&s5, &ps6}  // ERROR "&ps6 escapes to heap$" "&s5 escapes to heap$" "tUPiSPPic &U literal does not escape$"
	v := &V{u1, u2, &u3} // ERROR "tUPiSPPic &V literal does not escape$" "tUPiSPPic &u3 does not escape$"
	Ssink = v.UPiSPPic() // Ssink = *&ps4 = &s4 (only &s4 really escapes)
}

// BAD: need fine-grained (field-sensitive) analysis to avoid spurious escape of all but &s4
func tUPiSPPid() {
	s1 := "ant"
	s2 := "bat"
	s3 := "cat"
	s4 := "dog"          // ERROR "moved to heap: s4$"
	s5 := "emu"          // ERROR "moved to heap: s5$"
	s6 := "fox"          // ERROR "moved to heap: s6$"
	ps2 := &s2           // ERROR "tUPiSPPid &s2 does not escape$"
	ps4 := &s4           // ERROR "&s4 escapes to heap$"
	ps6 := &s6           // ERROR "&s6 escapes to heap$" "moved to heap: ps6$"
	u1 := U{&s1, &ps2}   // ERROR "tUPiSPPid &ps2 does not escape$" "tUPiSPPid &s1 does not escape$"
	u2 := &U{&s3, &ps4}  // ERROR "tUPiSPPid &U literal does not escape$" "tUPiSPPid &ps4 does not escape$" "tUPiSPPid &s3 does not escape$"
	u3 := &U{&s5, &ps6}  // ERROR "&ps6 escapes to heap$" "&s5 escapes to heap$" "tUPiSPPid &U literal does not escape$"
	v := &V{u1, u2, &u3} // ERROR "tUPiSPPid &V literal does not escape$" "tUPiSPPid &u3 does not escape$"
	Ssink = v.UPiSPPid() // Ssink = *&ps4 = &s4 (only &s4 really escapes)
}

func (v V) UPPiSPPia() *string { // ERROR "leaking param: v to result ~r0 level=3$"
	return *(*v._upp)._spp
}

// This test isolates the one value that needs to escape, not because
// it distinguishes fields but because it knows that &s6 is the only
// value reachable by two indirects from v.
// The test depends on the level cap in the escape analysis tags
// being able to encode that fact.
func tUPPiSPPia() { // This test is sensitive to the level cap in function summary results.
	s1 := "ant"
	s2 := "bat"
	s3 := "cat"
	s4 := "dog"
	s5 := "emu"
	s6 := "fox"           // ERROR "moved to heap: s6$"
	ps2 := &s2            // ERROR "tUPPiSPPia &s2 does not escape$"
	ps4 := &s4            // ERROR "tUPPiSPPia &s4 does not escape$"
	ps6 := &s6            // ERROR "&s6 escapes to heap$"
	u1 := U{&s1, &ps2}    // ERROR "tUPPiSPPia &ps2 does not escape$" "tUPPiSPPia &s1 does not escape$"
	u2 := &U{&s3, &ps4}   // ERROR "tUPPiSPPia &U literal does not escape$" "tUPPiSPPia &ps4 does not escape$" "tUPPiSPPia &s3 does not escape$"
	u3 := &U{&s5, &ps6}   // ERROR "tUPPiSPPia &U literal does not escape$" "tUPPiSPPia &ps6 does not escape$" "tUPPiSPPia &s5 does not escape$"
	v := &V{u1, u2, &u3}  // ERROR "tUPPiSPPia &V literal does not escape$" "tUPPiSPPia &u3 does not escape$"
	Ssink = v.UPPiSPPia() // Ssink = *&ps6 = &s6 (only &s6 really escapes)
}
