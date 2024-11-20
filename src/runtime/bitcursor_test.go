// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime_test

import (
	. "runtime"
	"testing"
)

func TestBitCursor(t *testing.T) {
	ones := [5]byte{0xff, 0xff, 0xff, 0xff, 0xff}
	zeros := [5]byte{0, 0, 0, 0, 0}

	for start := uintptr(0); start < 16; start++ {
		for end := start + 1; end < 32; end++ {
			buf := zeros
			NewBitCursor(&buf[0]).Offset(start).Write(&ones[0], end-start)

			for i := uintptr(0); i < uintptr(len(buf)*8); i++ {
				bit := buf[i/8] >> (i % 8) & 1
				if bit == 0 && i >= start && i < end {
					t.Errorf("bit %d not set in [%d:%d]", i, start, end)
				}
				if bit == 1 && (i < start || i >= end) {
					t.Errorf("bit %d is set outside [%d:%d]", i, start, end)
				}
			}
		}
	}

	for start := uintptr(0); start < 16; start++ {
		for end := start + 1; end < 32; end++ {
			buf := ones
			NewBitCursor(&buf[0]).Offset(start).Write(&zeros[0], end-start)

			for i := uintptr(0); i < uintptr(len(buf)*8); i++ {
				bit := buf[i/8] >> (i % 8) & 1
				if bit == 1 && i >= start && i < end {
					t.Errorf("bit %d not cleared in [%d:%d]", i, start, end)
				}
				if bit == 0 && (i < start || i >= end) {
					t.Errorf("bit %d cleared outside [%d:%d]", i, start, end)
				}
			}
		}
	}
}
