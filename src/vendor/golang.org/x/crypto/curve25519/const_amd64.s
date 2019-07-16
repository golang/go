// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This code was translated into a form compatible with 6a from the public
// domain sources in SUPERCOP: https://bench.cr.yp.to/supercop.html

// +build amd64,!gccgo,!appengine

// These constants cannot be encoded in non-MOVQ immediates.
// We access them directly from memory instead.

DATA ·_121666_213(SB)/8, $996687872
GLOBL ·_121666_213(SB), 8, $8

DATA ·_2P0(SB)/8, $0xFFFFFFFFFFFDA
GLOBL ·_2P0(SB), 8, $8

DATA ·_2P1234(SB)/8, $0xFFFFFFFFFFFFE
GLOBL ·_2P1234(SB), 8, $8
