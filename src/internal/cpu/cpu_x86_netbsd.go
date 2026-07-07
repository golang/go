// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (386 || amd64) && netbsd

package cpu

func osInit() {
	// NetBSD corrupts avx registers when receiving signals.
	// See issue 80285.
	// TODO: when NetBSD fixes the bug, add a version
	// check and skip here.
	X86.HasAVX = false
	X86.HasAVX2 = false
	// Set these also, just to be safe
	X86.HasAVXVNNI = false
	X86.HasAVX512 = false
	X86.HasAVX512F = false
	X86.HasAVX512CD = false
	X86.HasAVX512BW = false
	X86.HasAVX512DQ = false
	X86.HasAVX512VL = false
	X86.HasAVX512GFNI = false
	X86.HasAVX512VAES = false
	X86.HasAVX512VNNI = false
	X86.HasAVX512VBMI = false
	X86.HasAVX512VBMI2 = false
	X86.HasAVX512BITALG = false
	X86.HasAVX512VPOPCNTDQ = false
	X86.HasAVX512VPCLMULQDQ = false
}
