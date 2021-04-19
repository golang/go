// Copyright 2014 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ppc64asm

import (
	"testing"
)

func panicOrNot(f func()) (panicked bool) {
	defer func() {
		if err := recover(); err != nil {
			panicked = true
		}
	}()
	f()
	return false
}

func TestBitField(t *testing.T) {
	var tests = []struct {
		b    BitField
		i    uint32 // input
		u    uint32 // unsigned output
		s    int32  // signed output
		fail bool   // if the check should panic
	}{
		{BitField{0, 0}, 0, 0, 0, true},
		{BitField{31, 2}, 0, 0, 0, true},
		{BitField{31, 1}, 1, 1, -1, false},
		{BitField{29, 2}, 0 << 1, 0, 0, false},
		{BitField{29, 2}, 1 << 1, 1, 1, false},
		{BitField{29, 2}, 2 << 1, 2, -2, false},
		{BitField{29, 2}, 3 << 1, 3, -1, false},
		{BitField{0, 32}, 1<<32 - 1, 1<<32 - 1, -1, false},
		{BitField{16, 3}, 1 << 15, 4, -4, false},
	}
	for i, tst := range tests {
		var (
			ou uint32
			os int32
		)
		failed := panicOrNot(func() {
			ou = tst.b.Parse(tst.i)
			os = tst.b.ParseSigned(tst.i)
		})
		if failed != tst.fail {
			t.Errorf("case %d: %v: fail test failed, got %v, expected %v", i, tst.b, failed, tst.fail)
			continue
		}
		if ou != tst.u {
			t.Errorf("case %d: %v.Parse(%d) returned %d, expected %d", i, tst.b, tst.i, ou, tst.u)
			continue
		}
		if os != tst.s {
			t.Errorf("case %d: %v.ParseSigned(%d) returned %d, expected %d", i, tst.b, tst.i, os, tst.s)
		}
	}
}
