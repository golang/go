// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package riscv

import (
	"fmt"
	"testing"
)

func TestSplitShiftConst(t *testing.T) {
	tests := []struct {
		v       int64
		wantImm int64
		wantLsh int
		wantRsh int
		wantOk  bool
	}{
		{0x100000000, 1, 32, 0, true},
		{0xfffff001, 0, 0, 0, false},
		{0xfffff801, -2047, 32, 32, true},
		{0xfffffff1, -15, 32, 32, true},
		{0xffffffff, -1, 0, 32, true},
		{0xfffffffe, 0x7fffffff, 1, 0, true},
		{0xfffffffffffda, -19, 13, 12, true},
		{0xfffffffffffde, -17, 13, 12, true},
		{0x000003ffffffffff, -1, 0, 22, true},
		{0x0007ffffffffffff, -1, 0, 13, true},
		{0x7fffffff00000000, 0x7fffffff, 32, 0, true},
		{0x7fffffffffffffff, -1, 0, 1, true},
		{0x7f7f7f7f7f7f7f7f, 0, 0, 0, false},
		{0x0080000010000000, 0x8000001, 28, 0, true},
		{0x0abcdabcd0000000, 0, 0, 0, false},
		{-4503599610593281, 0, 0, 0, false}, // 0x8abcdabcd0000000
		{-7543254330000000, 0, 0, 0, false}, // 0xfff0000000ffffff
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("0x%x", test.v), func(t *testing.T) {
			c, l, r, ok := splitShiftConst(test.v)

			if got, want := c, test.wantImm; got != want {
				t.Errorf("Got immediate %d, want %d", got, want)
			}
			if got, want := l, test.wantLsh; got != want {
				t.Errorf("Got left shift %d, want %d", got, want)
			}
			if got, want := r, test.wantRsh; got != want {
				t.Errorf("Got right shift %d, want %d", got, want)
			}
			switch {
			case !ok && test.wantOk:
				t.Error("Failed to split shift constant, want success")
			case ok && !test.wantOk:
				t.Error("Successfully split shift constant, want failure")
			}
			if !ok || ok != test.wantOk {
				return
			}

			// Reconstruct as either a 12 bit or 32 bit signed constant.
			s := 64 - 12
			v := int64((uint64(((c << s) >> s)) << l) >> r)
			if test.wantImm != ((test.wantImm << s) >> s) {
				v = int64((uint64(int32(test.wantImm)) << l) >> r)
			}
			if v != test.v {
				t.Errorf("Got v = %d (%x), want v = %d (%x)", v, v, test.v, test.v)
			}
		})
	}
}
