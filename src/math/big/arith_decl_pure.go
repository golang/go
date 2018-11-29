// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build math_big_pure_go

package big

func mulWW(x, y Word) (z1, z0 Word) {
	return mulWW_g(x, y)
}

func divWW(x1, x0, y Word) (q, r Word) {
	return divWW_g(x1, x0, y)
}

func addVV(z, x, y []Word) (c Word) {
	return addVV_g(z, x, y)
}

func subVV(z, x, y []Word) (c Word) {
	return subVV_g(z, x, y)
}

func addVW(z, x []Word, y Word) (c Word) {
	return addVW_g(z, x, y)
}

func subVW(z, x []Word, y Word) (c Word) {
	return subVW_g(z, x, y)
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

func addMulVVW(z, x []Word, y Word) (c Word) {
	return addMulVVW_g(z, x, y)
}

func divWVW(z []Word, xn Word, x []Word, y Word) (r Word) {
	return divWVW_g(z, xn, x, y)
}
