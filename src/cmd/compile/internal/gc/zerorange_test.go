// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gc

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
