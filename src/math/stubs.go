// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !s390x

// This is a large group of functions that most architectures don't
// implement in assembly.

package math

const haveArchAcos = false

func archAcos(x float64) float64 {
	panic("not implemented")
}

const haveArchAcosh = false

func archAcosh(x float64) float64 {
	panic("not implemented")
}

const haveArchAsin = false

func archAsin(x float64) float64 {
	panic("not implemented")
}

const haveArchAsinh = false

func archAsinh(x float64) float64 {
	panic("not implemented")
}

const haveArchAtan = false

func archAtan(x float64) float64 {
	panic("not implemented")
}

const haveArchAtan2 = false

func archAtan2(y, x float64) float64 {
	panic("not implemented")
}

const haveArchAtanh = false

func archAtanh(x float64) float64 {
	panic("not implemented")
}

const haveArchCbrt = false

func archCbrt(x float64) float64 {
	panic("not implemented")
}

const haveArchCos = false

func archCos(x float64) float64 {
	panic("not implemented")
}

const haveArchCosh = false

func archCosh(x float64) float64 {
	panic("not implemented")
}

const haveArchErf = false

func archErf(x float64) float64 {
	panic("not implemented")
}

const haveArchErfc = false

func archErfc(x float64) float64 {
	panic("not implemented")
}

const haveArchExpm1 = false

func archExpm1(x float64) float64 {
	panic("not implemented")
}

const haveArchFrexp = false

func archFrexp(x float64) (float64, int) {
	panic("not implemented")
}

const haveArchLdexp = false

func archLdexp(frac float64, exp int) float64 {
	panic("not implemented")
}

const haveArchLog10 = false

func archLog10(x float64) float64 {
	panic("not implemented")
}

const haveArchLog2 = false

func archLog2(x float64) float64 {
	panic("not implemented")
}

const haveArchLog1p = false

func archLog1p(x float64) float64 {
	panic("not implemented")
}

const haveArchMod = false

func archMod(x, y float64) float64 {
	panic("not implemented")
}

const haveArchPow = false

func archPow(x, y float64) float64 {
	panic("not implemented")
}

const haveArchRemainder = false

func archRemainder(x, y float64) float64 {
	panic("not implemented")
}

const haveArchSin = false

func archSin(x float64) float64 {
	panic("not implemented")
}

const haveArchSinh = false

func archSinh(x float64) float64 {
	panic("not implemented")
}

const haveArchTan = false

func archTan(x float64) float64 {
	panic("not implemented")
}

const haveArchTanh = false

func archTanh(x float64) float64 {
	panic("not implemented")
}
