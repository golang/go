// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64

import "testing"

func testvmovs() (r1, r2 uint64)
func testvmovd() (r1, r2 uint64)
func testvmovq() (r1, r2 uint64)

func TestVMOV(t *testing.T) {
	tests := []struct {
		op           string
		vmovFunc     func() (uint64, uint64)
		wantA, wantB uint64
	}{
		{"VMOVS", testvmovs, 0x80402010, 0},
		{"VMOVD", testvmovd, 0x7040201008040201, 0},
		{"VMOVQ", testvmovq, 0x7040201008040201, 0x3040201008040201},
	}
	for _, test := range tests {
		gotA, gotB := test.vmovFunc()
		if gotA != test.wantA || gotB != test.wantB {
			t.Errorf("%v: got: a=0x%x, b=0x%x, want: a=0x%x, b=0x%x", test.op, gotA, gotB, test.wantA, test.wantB)
		}
	}
}

func testmovk() uint64

// TestMOVK makes sure MOVK with a very large constant works. See issue 52261.
func TestMOVK(t *testing.T) {
	x := testmovk()
	want := uint64(40000 << 48)
	if x != want {
		t.Errorf("Got %x want %x\n", x, want)
	}
}
