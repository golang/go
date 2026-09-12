// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package arm64

import "testing"

func TestIsADRP(t *testing.T) {
	tests := []struct {
		name string
		insn uint32
		want bool
	}{
		{"adrp x0", 0x90000000, true},
		{"adrp x27", 0x9000001b, true},
		{"adrp x27 with imm", 0xd000a3fb, true},  // from real binary
		{"adrp x27 with imm2", 0xf000a47b, true}, // from real binary
		{"add x0, x0, #0", 0x91000000, false},
		{"ldr x1, [x27]", 0xf9400361, false},
		{"b #0", 0x14000000, false},
		{"nop", 0xd503201f, false},
		{"str x3, [x27, #1080]", 0xf9021f63, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isADRP(tt.insn); got != tt.want {
				t.Errorf("isADRP(%#x) = %v, want %v", tt.insn, got, tt.want)
			}
		})
	}
}

func TestIsBranch(t *testing.T) {
	tests := []struct {
		name string
		insn uint32
		want bool
	}{
		{"b #0", 0x14000000, true},
		{"bl #0", 0x94000000, true},
		{"cbz x0, #0", 0xb4000000, true},
		{"cbnz w0, #0", 0x35000000, true},
		{"b.eq #0", 0x54000000, true},
		{"tbz x0, #0, #0", 0x36000000, true},
		{"br x5", 0xd61f00a0, true},
		{"blr x5", 0xd63f00a0, true},
		{"ret", 0xd65f03c0, true},
		{"adrp x0", 0x90000000, false},
		{"ldr x1, [x27]", 0xf9400361, false},
		{"nop", 0xd503201f, false},
		{"add x0, x0, #1", 0x91000400, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isBranch(tt.insn); got != tt.want {
				t.Errorf("isBranch(%#x) = %v, want %v", tt.insn, got, tt.want)
			}
		})
	}
}

func TestIsLoadStoreRegisterUnsigned(t *testing.T) {
	tests := []struct {
		name string
		insn uint32
		want bool
	}{
		// From real erratum matches in elma binary:
		{"ldr w2, [x27, #608]", 0xb9426362, true},
		{"str x2, [x27, #1888]", 0xf903b362, true},
		{"ldr x1, [x27, #1856]", 0xf943a361, true},
		{"str x3, [x27, #1080]", 0xf9021f63, true},
		// Non-unsigned-immediate:
		{"ldp x5, x0, [x0, #64]", 0xa94400a5, false},
		{"ldxr x0, [x1]", 0xc85f7c20, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if got := isLoadStoreRegisterUnsigned(tt.insn); got != tt.want {
				t.Errorf("isLoadStoreRegisterUnsigned(%#x) = %v, want %v", tt.insn, got, tt.want)
			}
		})
	}
}

func TestIs843419Sequence(t *testing.T) {
	// Real erratum match from elma 1.6.0-hotfix binary:
	// 0x7faffc: d000a3fb  adrp x27, 0x1c78000
	// 0x7fb000: f943a361  ldr  x1, [x27, #1856]
	// 0x7fb004: d000a57b  adrp x27, 0x1ca9000
	// 0x7fb008: b9426362  ldr  w2, [x27, #608]
	match1 := []uint32{0xd000a3fb, 0xf943a361, 0xd000a57b, 0xb9426362}

	// Real erratum match from elma 1.6.0-hotfix binary:
	// 0x81cff8: f000a47b  adrp x27, 0x1cab000
	// 0x81cffc: f9021f63  str  x3, [x27, #1080]
	// 0x81d000: f000a31b  adrp x27, 0x1c80000
	// 0x81d004: f903b362  str  x2, [x27, #1888]
	match2 := []uint32{0xf000a47b, 0xf9021f63, 0xf000a31b, 0xf903b362}

	// Non-match: ADRP followed immediately by dependent load (no gap).
	// This was the writeBarrier example that the Google engineer correctly
	// identified as NOT matching the erratum.
	noMatch := []uint32{0xd000a3fb, 0xb9426362, 0x14000000, 0xd503201f}

	tests := []struct {
		name       string
		insns      []uint32
		pageOffset uint64
		want       bool
	}{
		{"real match 1 (4-insn at 0xFFC)", match1, 0xFFC, true},
		{"real match 2 (4-insn at 0xFF8)", match2, 0xFF8, true},
		{"no match: dependent load at pos2", noMatch, 0xFF8, false},
		// Note: is843419Sequence does NOT check the page offset itself.
		// The caller (erratum843419Check) filters by page offset before calling.
		// So this test verifies the instruction pattern only.
		{"3-insn variant at 0xFFC", []uint32{
			0x90000000 | 27, // adrp x27
			0x91000000,      // add x0, x0, #0 (load/store class? no — this is NOT ls class)
			0xb9400360,      // ldr w0, [x27]
		}, 0xFFC, false}, // add is not load/store class, so pos2 fails
		{"3-insn variant valid at 0xFFC", []uint32{
			0x90000000 | 27, // adrp x27
			0xf9400001,      // ldr x1, [x0] — load, does not write x27
			0xb9400360,      // ldr w0, [x27] — unsigned imm load, base=x27
		}, 0xFFC, true},
		{"4-insn with branch at pos3", []uint32{
			0x90000000 | 27, // adrp x27
			0xf9400001,      // ldr x1, [x0]
			0x14000010,      // b #64 — branch at pos3 blocks 4-insn variant
			0xb9400360,      // ldr w0, [x27]
		}, 0xFF8, false},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			readInsn := func(delta int) (uint32, bool) {
				idx := delta / 4
				if idx < 0 || idx >= len(tt.insns) {
					return 0, false
				}
				return tt.insns[idx], true
			}
			got := is843419Sequence(tt.insns[0], tt.pageOffset, readInsn)
			if got != tt.want {
				t.Errorf("is843419Sequence(pageOffset=%#x) = %v, want %v", tt.pageOffset, got, tt.want)
			}
		})
	}
}
