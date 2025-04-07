// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !math_big_pure_go

package big

import _ "unsafe" // for linkname

// implemented in arith_$GOARCH.s

// addVV should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/remyoudompheng/bigfft
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname addVV
//go:noescape
func addVV(z, x, y []Word) (c Word)

// subVV should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/remyoudompheng/bigfft
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname subVV
//go:noescape
func subVV(z, x, y []Word) (c Word)

// shlVU should be an internal detail (and a stale one at that),
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/remyoudompheng/bigfft
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname shlVU
func shlVU(z, x []Word, s uint) (c Word) {
	if s == 0 {
		copy(z, x)
		return 0
	}
	return lshVU(z, x, s)
}

// lshVU sets z = x<<s, returning the high bits c. 1 ≤ s ≤ _B-1.
//
//go:noescape
func lshVU(z, x []Word, s uint) (c Word)

// rshVU sets z = x>>s, returning the low bits c. 1 ≤ s ≤ _B-1.
//
//go:noescape
func rshVU(z, x []Word, s uint) (c Word)

// mulAddVWW should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/remyoudompheng/bigfft
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname mulAddVWW
//go:noescape
func mulAddVWW(z, x []Word, m, a Word) (c Word)

// addMulVVW should be an internal detail (and a stale one at that),
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/remyoudompheng/bigfft
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname addMulVVW
func addMulVVW(z, x []Word, y Word) (c Word) {
	return addMulVVWW(z, z, x, y, 0)
}

// addMulVVWW sets z = x+y*m+a.
//
//go:noescape
func addMulVVWW(z, x, y []Word, m, a Word) (c Word)
