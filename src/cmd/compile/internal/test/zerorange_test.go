// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"testing"
)

var glob = 3
var globp *int64

// Testing compilation of arch.ZeroRange of various sizes.

// By storing a pointer to an int64 output param in a global, the compiler must
// ensure that output param is allocated on the heap. Also, since there is a
// defer, the pointer to each output param must be zeroed in the prologue (see
// plive.go:epilogue()). So, we will get a block of one or more stack slots that
// need to be zeroed. Hence, we are testing compilation completes successfully when
// zerorange calls of various sizes (8-136 bytes) are generated. We are not
// testing runtime correctness (which is hard to do for the current uses of
// ZeroRange).

func TestZeroRange(t *testing.T) {
	testZeroRange8(t)
	testZeroRange16(t)
	testZeroRange32(t)
	testZeroRange64(t)
	testZeroRange136(t)
}

func testZeroRange8(t *testing.T) (r int64) {
	defer func() {
		glob = 4
	}()
	globp = &r
	return
}

func testZeroRange16(t *testing.T) (r, s int64) {
	defer func() {
		glob = 4
	}()
	globp = &r
	globp = &s
	return
}

func testZeroRange32(t *testing.T) (r, s, t2, u int64) {
	defer func() {
		glob = 4
	}()
	globp = &r
	globp = &s
	globp = &t2
	globp = &u
	return
}

func testZeroRange64(t *testing.T) (r, s, t2, u, v, w, x, y int64) {
	defer func() {
		glob = 4
	}()
	globp = &r
	globp = &s
	globp = &t2
	globp = &u
	globp = &v
	globp = &w
	globp = &x
	globp = &y
	return
}

func testZeroRange136(t *testing.T) (r, s, t2, u, v, w, x, y, r1, s1, t1, u1, v1, w1, x1, y1, z1 int64) {
	defer func() {
		glob = 4
	}()
	globp = &r
	globp = &s
	globp = &t2
	globp = &u
	globp = &v
	globp = &w
	globp = &x
	globp = &y
	globp = &r1
	globp = &s1
	globp = &t1
	globp = &u1
	globp = &v1
	globp = &w1
	globp = &x1
	globp = &y1
	globp = &z1
	return
}

type S struct {
	x [2]uint64
	p *uint64
	y [2]uint64
	q uint64
}

type M struct {
	x [8]uint64
	p *uint64
	y [8]uint64
	q uint64
}

type L struct {
	x [4096]uint64
	p *uint64
	y [4096]uint64
	q uint64
}

//go:noinline
func triggerZerorangeLarge(f, g, h uint64) (rv0 uint64) {
	ll := L{p: &f}
	da := f
	rv0 = f + g + h
	defer func(dl L, i uint64) {
		rv0 += dl.q + i
	}(ll, da)
	return rv0
}

//go:noinline
func triggerZerorangeMedium(f, g, h uint64) (rv0 uint64) {
	ll := M{p: &f}
	rv0 = f + g + h
	defer func(dm M, i uint64) {
		rv0 += dm.q + i
	}(ll, f)
	return rv0
}

//go:noinline
func triggerZerorangeSmall(f, g, h uint64) (rv0 uint64) {
	ll := S{p: &f}
	rv0 = f + g + h
	defer func(ds S, i uint64) {
		rv0 += ds.q + i
	}(ll, f)
	return rv0
}

// This test was created as a follow up to issue #45372, to help
// improve coverage of the compiler's arch-specific "zerorange"
// function, which is invoked to zero out ambiguously live portions of
// the stack frame in certain specific circumstances.
//
// In the current compiler implementation, for zerorange to be
// invoked, we need to have an ambiguously live variable that needs
// zeroing. One way to trigger this is to have a function with an
// open-coded defer, where the opendefer function has an argument that
// contains a pointer (this is what's used below).
//
// At the moment this test doesn't do any specific checking for
// code sequence, or verification that things were properly set to zero,
// this seems as though it would be too tricky and would result
// in a "brittle" test.
//
// The small/medium/large scenarios below are inspired by the amd64
// implementation of zerorange, which generates different code
// depending on the size of the thing that needs to be zeroed out
// (I've verified at the time of the writing of this test that it
// exercises the various cases).
//
func TestZerorange45372(t *testing.T) {
	if r := triggerZerorangeLarge(101, 303, 505); r != 1010 {
		t.Errorf("large: wanted %d got %d", 1010, r)
	}
	if r := triggerZerorangeMedium(101, 303, 505); r != 1010 {
		t.Errorf("medium: wanted %d got %d", 1010, r)
	}
	if r := triggerZerorangeSmall(101, 303, 505); r != 1010 {
		t.Errorf("small: wanted %d got %d", 1010, r)
	}

}
