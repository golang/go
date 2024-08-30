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

// addVW should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/remyoudompheng/bigfft
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname addVW
//go:noescape
func addVW(z, x []Word, y Word) (c Word)

// subVW should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/remyoudompheng/bigfft
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname subVW
//go:noescape
func subVW(z, x []Word, y Word) (c Word)

// shlVU should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/remyoudompheng/bigfft
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname shlVU
//go:noescape
func shlVU(z, x []Word, s uint) (c Word)

//go:noescape
func shrVU(z, x []Word, s uint) (c Word)

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
func mulAddVWW(z, x []Word, y, r Word) (c Word)

// addMulVVW should be an internal detail,
// but widely used packages access it using linkname.
// Notable members of the hall of shame include:
//   - github.com/remyoudompheng/bigfft
//
// Do not remove or change the type signature.
// See go.dev/issue/67401.
//
//go:linkname addMulVVW
//go:noescape
func addMulVVW(z, x []Word, y Word) (c Word)
