// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build math_big_pure_go

package big

func addVV(z, x, y []Word) (c Word) {
	return addVV_g(z, x, y)
}

func subVV(z, x, y []Word) (c Word) {
	return subVV_g(z, x, y)
}

func addVW(z, x []Word, y Word) (c Word) {
	// TODO: remove indirect function call when golang.org/issue/30548 is fixed
	fn := addVW_g
	if len(z) > 32 {
		fn = addVWlarge
	}
	return fn(z, x, y)
}

func subVW(z, x []Word, y Word) (c Word) {
	// TODO: remove indirect function call when golang.org/issue/30548 is fixed
	fn := subVW_g
	if len(z) > 32 {
		fn = subVWlarge
	}
	return fn(z, x, y)
}

func shlVU(z, x []Word, s uint) (c Word) {
	return shlVU_g(z, x, s)
}

func shrVU(z, x []Word, s uint) (c Word) {
	return shrVU_g(z, x, s)
}

func mulAddVWW(z, x []Word, y, r Word) (c Word) {
	return mulAddVWW_g(z, x, y, r)
}

func addMulVVWW(z, x, y []Word, m, a Word) (c Word) {
	return addMulVVWW_g(z, x, y, m, a)
}
