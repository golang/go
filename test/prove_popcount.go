// errorcheck -0 -d=ssa/prove/debug=1

//go:build amd64.v3 || arm64

// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// FIXME(@Jorropo): this file exists because I havn't yet bothered to
// make prove work on the pure go function call fallback.
// My idea was to wait until CL 637936 is merged, then we can always emit
// the PopCount SSA operation and translate them to pure function calls
// in late-opt.

package main

import "math/bits"

func onesCountsBounds(x uint64, ensureAllBranchesCouldHappen func() bool) int {
	z := bits.OnesCount64(x)
	if ensureAllBranchesCouldHappen() && z > 64 { // ERROR "Disproved Less64$"
		return 42
	}
	if ensureAllBranchesCouldHappen() && z <= 64 { // ERROR "Proved Leq64$"
		return 4242
	}
	if ensureAllBranchesCouldHappen() && z < 0 { // ERROR "Disproved Less64$"
		return 424242
	}
	if ensureAllBranchesCouldHappen() && z >= 0 { // ERROR "Proved Leq64$"
		return 42424242
	}
	return z
}

func onesCountsTight(x uint64, ensureAllBranchesCouldHappen func() bool) int {
	const maxv = 0xff0f
	const minv = 0xff00
	x = max(x, minv)
	x = min(x, maxv)

	z := bits.OnesCount64(x)

	if ensureAllBranchesCouldHappen() && z > bits.OnesCount64(maxv) { // ERROR "Disproved Less64$"
		return 42
	}
	if ensureAllBranchesCouldHappen() && z <= bits.OnesCount64(maxv) { // ERROR "Proved Leq64$"
		return 4242
	}
	if ensureAllBranchesCouldHappen() && z < bits.OnesCount64(minv) { // ERROR "Disproved Less64$"
		return 424242
	}
	if ensureAllBranchesCouldHappen() && z >= bits.OnesCount64(minv) { // ERROR "Proved Leq64$"
		return 42424242
	}
	return z
}

func main() {
}
