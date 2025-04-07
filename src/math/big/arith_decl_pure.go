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

func lshVU(z, x []Word, s uint) (c Word) {
	return lshVU_g(z, x, s)
}

func rshVU(z, x []Word, s uint) (c Word) {
	return rshVU_g(z, x, s)
}

func mulAddVWW(z, x []Word, y, r Word) (c Word) {
	return mulAddVWW_g(z, x, y, r)
}

func addMulVVWW(z, x, y []Word, m, a Word) (c Word) {
	return addMulVVWW_g(z, x, y, m, a)
}
