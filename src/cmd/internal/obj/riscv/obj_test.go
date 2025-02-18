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
		wantOk  bool
	}{
		{0x100000000, 1, 32, true},
		{0xfffff001, 0, 0, false},
		{0xfffff801, 0, 0, false},
		{0xfffffff1, 0, 0, false},
		{0xffffffff, 0, 0, false},
		{0xfffffffe, 0x7fffffff, 1, true},
		{0xfffffffffffda, 0, 0, false},
		{0xfffffffffffde, 0, 0, false},
		{0x000003ffffffffff, 0, 0, false},
		{0x0007ffffffffffff, 0, 0, false},
		{0x7fffffff00000000, 0x7fffffff, 32, true},
		{0x7fffffffffffffff, 0, 0, false},
		{0x7f7f7f7f7f7f7f7f, 0, 0, false},
		{0x0080000010000000, 0x8000001, 28, true},
		{0x0abcdabcd0000000, 0, 0, false},
		{-4503599610593281, 0, 0, false}, // 0x8abcdabcd0000000
		{-7543254330000000, 0, 0, false}, // 0xfff0000000ffffff
	}
	for _, test := range tests {
		t.Run(fmt.Sprintf("0x%x", test.v), func(t *testing.T) {
			c, l, ok := splitShiftConst(test.v)

			if got, want := c, test.wantImm; got != want {
				t.Errorf("Got immediate %d, want %d", got, want)
			}
			if got, want := l, test.wantLsh; got != want {
				t.Errorf("Got left shift %d, want %d", got, want)
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

			// Reconstruct as a 32 bit signed constant.
			v := int64(uint64(int32(test.wantImm)) << l)
			if v != test.v {
				t.Errorf("Got v = %d (%x), want v = %d (%x)", v, v, test.v, test.v)
			}
		})
	}
}
