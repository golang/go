// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (aix || linux) && (ppc64 || ppc64le)

package bytealg

import "internal/cpu"

const MaxBruteForce = 16

var SupportsPower9 = cpu.PPC64.IsPOWER9

func init() {
	MaxLen = 32
}

// Cutover reports the number of failures of IndexByte we should tolerate
// before switching over to Index.
// n is the number of bytes processed so far.
// See the bytes.Index implementation for details.
func Cutover(n int) int {
	// 1 error per 8 characters, plus a few slop to start.
	return (n + 16) / 8
}
