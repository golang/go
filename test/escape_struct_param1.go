// errorcheck -0 -m -l

// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Test escape analysis for *struct function parameters.
// Note companion strict_param2 checks struct function parameters with similar tests.

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

func (u *U) SP() *string { // ERROR "leaking param: u to result ~r0 level=1$"
	return u._sp
}

func (u *U) SPP() **string { // ERROR "leaking param: u to result ~r0 level=1$"
	return u._spp
}

func (u *U) SPPi() *string { // ERROR "leaking param: u to result ~r0 level=2$"
	return *u._spp
}

func tSPPi() {
	s := "cat" // ERROR "moved to heap: s$"
	ps := &s
	pps := &ps
	pu := &U{ps, pps} // ERROR "&U{...} does not escape$"
	Ssink = pu.SPPi()
}

func tiSPP() {
	s := "cat" // ERROR "moved to heap: s$"
	ps := &s
	pps := &ps
	pu := &U{ps, pps} // ERROR "&U{...} does not escape$"
	Ssink = *pu.SPP()
}

// BAD: need fine-grained (field-sensitive) analysis to avoid spurious escape of ps
func tSP() {
	s := "cat" // ERROR "moved to heap: s$"
	ps := &s   // ERROR "moved to heap: ps$"
	pps := &ps
	pu := &U{ps, pps} // ERROR "&U{...} does not escape$"
	Ssink = pu.SP()
}

func (v *V) u() U { // ERROR "leaking param: v to result ~r0 level=1$"
	return v._u
}

func (v *V) UP() *U { // ERROR "leaking param: v to result ~r0 level=1$"
	return v._up
}

func (v *V) UPP() **U { // ERROR "leaking param: v to result ~r0 level=1$"
	return v._upp
}

func (v *V) UPPia() *U { // ERROR "leaking param: v to result ~r0 level=2$"
	return *v._upp
}

func (v *V) UPPib() *U { // ERROR "leaking param: v to result ~r0 level=2$"
	return *v.UPP()
}

func (v *V) USPa() *string { // ERROR "leaking param: v to result ~r0 level=1$"
	return v._u._sp
}

func (v *V) USPb() *string { // ERROR "leaking param: v to result ~r0 level=1$"
	return v.u()._sp
}

func (v *V) USPPia() *string { // ERROR "leaking param: v to result ~r0 level=2$"
	return *v._u._spp
}

func (v *V) USPPib() *string { // ERROR "leaking param: v to result ~r0 level=2$"
	return v._u.SPPi()
}

func (v *V) UPiSPa() *string { // ERROR "leaking param: v to result ~r0 level=2$"
	return v._up._sp
}

func (v *V) UPiSPb() *string { // ERROR "leaking param: v to result ~r0 level=2$"
	return v._up.SP()
}

func (v *V) UPiSPc() *string { // ERROR "leaking param: v to result ~r0 level=2$"
	return v.UP()._sp
}

func (v *V) UPiSPd() *string { // ERROR "leaking param: v to result ~r0 level=2$"
	return v.UP().SP()
}

// BAD: need fine-grained (field-sensitive) analysis to avoid spurious escape of all but &s3
func tUPiSPa() {
	s1 := "ant"
	s2 := "bat" // ERROR "moved to heap: s2$"
	s3 := "cat" // ERROR "moved to heap: s3$"
	s4 := "dog" // ERROR "moved to heap: s4$"
	s5 := "emu" // ERROR "moved to heap: s5$"
	s6 := "fox" // ERROR "moved to heap: s6$"
	ps2 := &s2
	ps4 := &s4 // ERROR "moved to heap: ps4$"
	ps6 := &s6 // ERROR "moved to heap: ps6$"
	u1 := U{&s1, &ps2}
	u2 := &U{&s3, &ps4}  // ERROR "&U{...} does not escape$"
	u3 := &U{&s5, &ps6}  // ERROR "&U{...} escapes to heap$"
	v := &V{u1, u2, &u3} // ERROR "&V{...} does not escape$"
	Ssink = v.UPiSPa()   // Ssink = &s3 (only &s3 really escapes)
}

// BAD: need fine-grained (field-sensitive) analysis to avoid spurious escape of all but &s3
func tUPiSPb() {
	s1 := "ant"
	s2 := "bat" // ERROR "moved to heap: s2$"
	s3 := "cat" // ERROR "moved to heap: s3$"
	s4 := "dog" // ERROR "moved to heap: s4$"
	s5 := "emu" // ERROR "moved to heap: s5$"
	s6 := "fox" // ERROR "moved to heap: s6$"
	ps2 := &s2
	ps4 := &s4 // ERROR "moved to heap: ps4$"
	ps6 := &s6 // ERROR "moved to heap: ps6$"
	u1 := U{&s1, &ps2}
	u2 := &U{&s3, &ps4}  // ERROR "&U{...} does not escape$"
	u3 := &U{&s5, &ps6}  // ERROR "&U{...} escapes to heap$"
	v := &V{u1, u2, &u3} // ERROR "&V{...} does not escape$"
	Ssink = v.UPiSPb()   // Ssink = &s3 (only &s3 really escapes)
}

// BAD: need fine-grained (field-sensitive) analysis to avoid spurious escape of all but &s3
func tUPiSPc() {
	s1 := "ant"
	s2 := "bat" // ERROR "moved to heap: s2$"
	s3 := "cat" // ERROR "moved to heap: s3$"
	s4 := "dog" // ERROR "moved to heap: s4$"
	s5 := "emu" // ERROR "moved to heap: s5$"
	s6 := "fox" // ERROR "moved to heap: s6$"
	ps2 := &s2
	ps4 := &s4 // ERROR "moved to heap: ps4$"
	ps6 := &s6 // ERROR "moved to heap: ps6$"
	u1 := U{&s1, &ps2}
	u2 := &U{&s3, &ps4}  // ERROR "&U{...} does not escape$"
	u3 := &U{&s5, &ps6}  // ERROR "&U{...} escapes to heap$"
	v := &V{u1, u2, &u3} // ERROR "&V{...} does not escape$"
	Ssink = v.UPiSPc()   // Ssink = &s3 (only &s3 really escapes)
}

// BAD: need fine-grained (field-sensitive) analysis to avoid spurious escape of all but &s3
func tUPiSPd() {
	s1 := "ant"
	s2 := "bat" // ERROR "moved to heap: s2$"
	s3 := "cat" // ERROR "moved to heap: s3$"
	s4 := "dog" // ERROR "moved to heap: s4$"
	s5 := "emu" // ERROR "moved to heap: s5$"
	s6 := "fox" // ERROR "moved to heap: s6$"
	ps2 := &s2
	ps4 := &s4 // ERROR "moved to heap: ps4$"
	ps6 := &s6 // ERROR "moved to heap: ps6$"
	u1 := U{&s1, &ps2}
	u2 := &U{&s3, &ps4}  // ERROR "&U{...} does not escape$"
	u3 := &U{&s5, &ps6}  // ERROR "&U{...} escapes to heap$"
	v := &V{u1, u2, &u3} // ERROR "&V{...} does not escape$"
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
	s4 := "dog" // ERROR "moved to heap: s4$"
	s5 := "emu" // ERROR "moved to heap: s5$"
	s6 := "fox" // ERROR "moved to heap: s6$"
	ps2 := &s2
	ps4 := &s4
	ps6 := &s6 // ERROR "moved to heap: ps6$"
	u1 := U{&s1, &ps2}
	u2 := &U{&s3, &ps4}  // ERROR "&U{...} does not escape$"
	u3 := &U{&s5, &ps6}  // ERROR "&U{...} does not escape$"
	v := &V{u1, u2, &u3} // ERROR "&V{...} does not escape$"
	Ssink = v.UPiSPPia() // Ssink = *&ps4 = &s4 (only &s4 really escapes)
}

// BAD: need fine-grained (field-sensitive) analysis to avoid spurious escape of all but &s4
func tUPiSPPib() {
	s1 := "ant"
	s2 := "bat"
	s3 := "cat"
	s4 := "dog" // ERROR "moved to heap: s4$"
	s5 := "emu" // ERROR "moved to heap: s5$"
	s6 := "fox" // ERROR "moved to heap: s6$"
	ps2 := &s2
	ps4 := &s4
	ps6 := &s6 // ERROR "moved to heap: ps6$"
	u1 := U{&s1, &ps2}
	u2 := &U{&s3, &ps4}  // ERROR "&U{...} does not escape$"
	u3 := &U{&s5, &ps6}  // ERROR "&U{...} does not escape$"
	v := &V{u1, u2, &u3} // ERROR "&V{...} does not escape$"
	Ssink = v.UPiSPPib() // Ssink = *&ps4 = &s4 (only &s4 really escapes)
}

// BAD: need fine-grained (field-sensitive) analysis to avoid spurious escape of all but &s4
func tUPiSPPic() {
	s1 := "ant"
	s2 := "bat"
	s3 := "cat"
	s4 := "dog" // ERROR "moved to heap: s4$"
	s5 := "emu" // ERROR "moved to heap: s5$"
	s6 := "fox" // ERROR "moved to heap: s6$"
	ps2 := &s2
	ps4 := &s4
	ps6 := &s6 // ERROR "moved to heap: ps6$"
	u1 := U{&s1, &ps2}
	u2 := &U{&s3, &ps4}  // ERROR "&U{...} does not escape$"
	u3 := &U{&s5, &ps6}  // ERROR "&U{...} does not escape$"
	v := &V{u1, u2, &u3} // ERROR "&V{...} does not escape$"
	Ssink = v.UPiSPPic() // Ssink = *&ps4 = &s4 (only &s4 really escapes)
}

// BAD: need fine-grained (field-sensitive) analysis to avoid spurious escape of all but &s4
func tUPiSPPid() {
	s1 := "ant"
	s2 := "bat"
	s3 := "cat"
	s4 := "dog" // ERROR "moved to heap: s4$"
	s5 := "emu" // ERROR "moved to heap: s5$"
	s6 := "fox" // ERROR "moved to heap: s6$"
	ps2 := &s2
	ps4 := &s4
	ps6 := &s6 // ERROR "moved to heap: ps6$"
	u1 := U{&s1, &ps2}
	u2 := &U{&s3, &ps4}  // ERROR "&U{...} does not escape$"
	u3 := &U{&s5, &ps6}  // ERROR "&U{...} does not escape$"
	v := &V{u1, u2, &u3} // ERROR "&V{...} does not escape$"
	Ssink = v.UPiSPPid() // Ssink = *&ps4 = &s4 (only &s4 really escapes)
}

func (v *V) UPPiSPPia() *string { // ERROR "leaking param: v to result ~r0 level=4$"
	return *(*v._upp)._spp
}

// This test isolates the one value that needs to escape, not because
// it distinguishes fields but because it knows that &s6 is the only
// value reachable by two indirects from v.
// The test depends on the level cap in the escape analysis tags
// being able to encode that fact.
func tUPPiSPPia() {
	s1 := "ant"
	s2 := "bat"
	s3 := "cat"
	s4 := "dog"
	s5 := "emu"
	s6 := "fox" // ERROR "moved to heap: s6$"
	ps2 := &s2
	ps4 := &s4
	ps6 := &s6
	u1 := U{&s1, &ps2}
	u2 := &U{&s3, &ps4}   // ERROR "&U{...} does not escape$"
	u3 := &U{&s5, &ps6}   // ERROR "&U{...} does not escape$"
	v := &V{u1, u2, &u3}  // ERROR "&V{...} does not escape$"
	Ssink = v.UPPiSPPia() // Ssink = *&ps6 = &s6 (only &s6 really escapes)
}
