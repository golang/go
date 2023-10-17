// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !math_big_pure_go

package big

// implemented in arith_$GOARCH.s

//go:noescape
func addVV(z, x, y []Word) (c Word)

//go:noescape
func subVV(z, x, y []Word) (c Word)

//go:noescape
func addVW(z, x []Word, y Word) (c Word)

//go:noescape
func subVW(z, x []Word, y Word) (c Word)

//go:noescape
func shlVU(z, x []Word, s uint) (c Word)

//go:noescape
func shrVU(z, x []Word, s uint) (c Word)

//go:noescape
func mulAddVWW(z, x []Word, y, r Word) (c Word)

//go:noescape
func addMulVVW(z, x []Word, y Word) (c Word)
