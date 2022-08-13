// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ssa

import "testing"

// We generate memmove for copy(x[1:], x[:]), however we may change it to OpMove,
// because size is known. Check that OpMove is alias-safe, or we did call memmove.
func TestMove(t *testing.T) {
	x := [...]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40}
	copy(x[1:], x[:])
	for i := 1; i < len(x); i++ {
		if int(x[i]) != i {
			t.Errorf("Memmove got converted to OpMove in alias-unsafe way. Got %d instead of %d in position %d", int(x[i]), i, i+1)
		}
	}
}

func TestMoveSmall(t *testing.T) {
	x := [...]byte{1, 2, 3, 4, 5, 6, 7}
	copy(x[1:], x[:])
	for i := 1; i < len(x); i++ {
		if int(x[i]) != i {
			t.Errorf("Memmove got converted to OpMove in alias-unsafe way. Got %d instead of %d in position %d", int(x[i]), i, i+1)
		}
	}
}

func TestSubFlags(t *testing.T) {
	if !subFlags32(0, 1).lt() {
		t.Errorf("subFlags32(0,1).lt() returned false")
	}
	if !subFlags32(0, 1).ult() {
		t.Errorf("subFlags32(0,1).ult() returned false")
	}
}

func TestIsPPC64WordRotateMask(t *testing.T) {
	tests := []struct {
		input    int64
		expected bool
	}{
		{0x00000001, true},
		{0x80000001, true},
		{0x80010001, false},
		{0xFFFFFFFA, false},
		{0xF0F0F0F0, false},
		{0xFFFFFFFD, true},
		{0x80000000, true},
		{0x00000000, false},
		{0xFFFFFFFF, true},
		{0x0000FFFF, true},
		{0xFF0000FF, true},
		{0x00FFFF00, true},
	}

	for _, v := range tests {
		if v.expected != isPPC64WordRotateMask(v.input) {
			t.Errorf("isPPC64WordRotateMask(0x%x) failed", v.input)
		}
	}
}

func TestEncodeDecodePPC64WordRotateMask(t *testing.T) {
	tests := []struct {
		rotate int64
		mask   uint64
		nbits,
		mb,
		me,
		encoded int64
	}{
		{1, 0x00000001, 32, 31, 31, 0x20011f20},
		{2, 0x80000001, 32, 31, 0, 0x20021f01},
		{3, 0xFFFFFFFD, 32, 31, 29, 0x20031f1e},
		{4, 0x80000000, 32, 0, 0, 0x20040001},
		{5, 0xFFFFFFFF, 32, 0, 31, 0x20050020},
		{6, 0x0000FFFF, 32, 16, 31, 0x20061020},
		{7, 0xFF0000FF, 32, 24, 7, 0x20071808},
		{8, 0x00FFFF00, 32, 8, 23, 0x20080818},

		{9, 0x0000000000FFFF00, 64, 40, 55, 0x40092838},
		{10, 0xFFFF000000000000, 64, 0, 15, 0x400A0010},
		{10, 0xFFFF000000000001, 64, 63, 15, 0x400A3f10},
	}

	for i, v := range tests {
		result := encodePPC64RotateMask(v.rotate, int64(v.mask), v.nbits)
		if result != v.encoded {
			t.Errorf("encodePPC64RotateMask(%d,0x%x,%d) = 0x%x, expected 0x%x", v.rotate, v.mask, v.nbits, result, v.encoded)
		}
		rotate, mb, me, mask := DecodePPC64RotateMask(result)
		if rotate != v.rotate || mb != v.mb || me != v.me || mask != v.mask {
			t.Errorf("DecodePPC64Failure(Test %d) got (%d, %d, %d, %x) expected (%d, %d, %d, %x)", i, rotate, mb, me, mask, v.rotate, v.mb, v.me, v.mask)
		}
	}
}

func TestMergePPC64ClrlsldiSrw(t *testing.T) {
	tests := []struct {
		clrlsldi int32
		srw      int64
		valid    bool
		rotate   int64
		mask     uint64
	}{
		// ((x>>4)&0xFF)<<4
		{newPPC64ShiftAuxInt(4, 56, 63, 64), 4, true, 0, 0xFF0},
		// ((x>>4)&0xFFFF)<<4
		{newPPC64ShiftAuxInt(4, 48, 63, 64), 4, true, 0, 0xFFFF0},
		// ((x>>4)&0xFFFF)<<17
		{newPPC64ShiftAuxInt(17, 48, 63, 64), 4, false, 0, 0},
		// ((x>>4)&0xFFFF)<<16
		{newPPC64ShiftAuxInt(16, 48, 63, 64), 4, true, 12, 0xFFFF0000},
		// ((x>>32)&0xFFFF)<<17
		{newPPC64ShiftAuxInt(17, 48, 63, 64), 32, false, 0, 0},
	}
	for i, v := range tests {
		result := mergePPC64ClrlsldiSrw(int64(v.clrlsldi), v.srw)
		if v.valid && result == 0 {
			t.Errorf("mergePPC64ClrlsldiSrw(Test %d) did not merge", i)
		} else if !v.valid && result != 0 {
			t.Errorf("mergePPC64ClrlsldiSrw(Test %d) should return 0", i)
		} else if r, _, _, m := DecodePPC64RotateMask(result); v.rotate != r || v.mask != m {
			t.Errorf("mergePPC64ClrlsldiSrw(Test %d) got (%d,0x%x) expected (%d,0x%x)", i, r, m, v.rotate, v.mask)
		}
	}
}

func TestMergePPC64ClrlsldiRlwinm(t *testing.T) {
	tests := []struct {
		clrlsldi int32
		rlwinm   int64
		valid    bool
		rotate   int64
		mask     uint64
	}{
		// ((x<<4)&0xFF00)<<4
		{newPPC64ShiftAuxInt(4, 56, 63, 64), encodePPC64RotateMask(4, 0xFF00, 32), false, 0, 0},
		// ((x>>4)&0xFF)<<4
		{newPPC64ShiftAuxInt(4, 56, 63, 64), encodePPC64RotateMask(28, 0x0FFFFFFF, 32), true, 0, 0xFF0},
		// ((x>>4)&0xFFFF)<<4
		{newPPC64ShiftAuxInt(4, 48, 63, 64), encodePPC64RotateMask(28, 0xFFFF, 32), true, 0, 0xFFFF0},
		// ((x>>4)&0xFFFF)<<17
		{newPPC64ShiftAuxInt(17, 48, 63, 64), encodePPC64RotateMask(28, 0xFFFF, 32), false, 0, 0},
		// ((x>>4)&0xFFFF)<<16
		{newPPC64ShiftAuxInt(16, 48, 63, 64), encodePPC64RotateMask(28, 0xFFFF, 32), true, 12, 0xFFFF0000},
		// ((x>>4)&0xF000FFFF)<<16
		{newPPC64ShiftAuxInt(16, 48, 63, 64), encodePPC64RotateMask(28, 0xF000FFFF, 32), true, 12, 0xFFFF0000},
	}
	for i, v := range tests {
		result := mergePPC64ClrlsldiRlwinm(v.clrlsldi, v.rlwinm)
		if v.valid && result == 0 {
			t.Errorf("mergePPC64ClrlsldiRlwinm(Test %d) did not merge", i)
		} else if !v.valid && result != 0 {
			t.Errorf("mergePPC64ClrlsldiRlwinm(Test %d) should return 0", i)
		} else if r, _, _, m := DecodePPC64RotateMask(result); v.rotate != r || v.mask != m {
			t.Errorf("mergePPC64ClrlsldiRlwinm(Test %d) got (%d,0x%x) expected (%d,0x%x)", i, r, m, v.rotate, v.mask)
		}
	}
}

func TestMergePPC64SldiSrw(t *testing.T) {
	tests := []struct {
		sld    int64
		srw    int64
		valid  bool
		rotate int64
		mask   uint64
	}{
		{4, 4, true, 0, 0xFFFFFFF0},
		{4, 8, true, 28, 0x0FFFFFF0},
		{0, 0, true, 0, 0xFFFFFFFF},
		{8, 4, false, 0, 0},
		{0, 32, false, 0, 0},
		{0, 31, true, 1, 0x1},
		{31, 31, true, 0, 0x80000000},
		{32, 32, false, 0, 0},
	}
	for i, v := range tests {
		result := mergePPC64SldiSrw(v.sld, v.srw)
		if v.valid && result == 0 {
			t.Errorf("mergePPC64SldiSrw(Test %d) did not merge", i)
		} else if !v.valid && result != 0 {
			t.Errorf("mergePPC64SldiSrw(Test %d) should return 0", i)
		} else if r, _, _, m := DecodePPC64RotateMask(result); v.rotate != r || v.mask != m {
			t.Errorf("mergePPC64SldiSrw(Test %d) got (%d,0x%x) expected (%d,0x%x)", i, r, m, v.rotate, v.mask)
		}
	}
}

func TestMergePPC64AndSrwi(t *testing.T) {
	tests := []struct {
		and    int64
		srw    int64
		valid  bool
		rotate int64
		mask   uint64
	}{
		{0x000000FF, 8, true, 24, 0xFF},
		{0xF00000FF, 8, true, 24, 0xFF},
		{0x0F0000FF, 4, false, 0, 0},
		{0x00000000, 4, false, 0, 0},
		{0xF0000000, 4, false, 0, 0},
		{0xF0000000, 32, false, 0, 0},
		{0xFFFFFFFF, 0, true, 0, 0xFFFFFFFF},
	}
	for i, v := range tests {
		result := mergePPC64AndSrwi(v.and, v.srw)
		if v.valid && result == 0 {
			t.Errorf("mergePPC64AndSrwi(Test %d) did not merge", i)
		} else if !v.valid && result != 0 {
			t.Errorf("mergePPC64AndSrwi(Test %d) should return 0", i)
		} else if r, _, _, m := DecodePPC64RotateMask(result); v.rotate != r || v.mask != m {
			t.Errorf("mergePPC64AndSrwi(Test %d) got (%d,0x%x) expected (%d,0x%x)", i, r, m, v.rotate, v.mask)
		}
	}
}
