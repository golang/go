// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

import "internal/cpu"

func expTrampolineSetup(x float64) float64
func expAsm(x float64) float64

func logTrampolineSetup(x float64) float64
func logAsm(x float64) float64

// Below here all functions are grouped in stubs.go for other
// architectures.

const haveArchLog10 = true

func archLog10(x float64) float64
func log10TrampolineSetup(x float64) float64
func log10Asm(x float64) float64

const haveArchCos = true

func archCos(x float64) float64
func cosTrampolineSetup(x float64) float64
func cosAsm(x float64) float64

const haveArchCosh = true

func archCosh(x float64) float64
func coshTrampolineSetup(x float64) float64
func coshAsm(x float64) float64

const haveArchSin = true

func archSin(x float64) float64
func sinTrampolineSetup(x float64) float64
func sinAsm(x float64) float64

const haveArchSinh = true

func archSinh(x float64) float64
func sinhTrampolineSetup(x float64) float64
func sinhAsm(x float64) float64

const haveArchTanh = true

func archTanh(x float64) float64
func tanhTrampolineSetup(x float64) float64
func tanhAsm(x float64) float64

const haveArchLog1p = true

func archLog1p(x float64) float64
func log1pTrampolineSetup(x float64) float64
func log1pAsm(x float64) float64

const haveArchAtanh = true

func archAtanh(x float64) float64
func atanhTrampolineSetup(x float64) float64
func atanhAsm(x float64) float64

const haveArchAcos = true

func archAcos(x float64) float64
func acosTrampolineSetup(x float64) float64
func acosAsm(x float64) float64

const haveArchAcosh = true

func archAcosh(x float64) float64
func acoshTrampolineSetup(x float64) float64
func acoshAsm(x float64) float64

const haveArchAsin = true

func archAsin(x float64) float64
func asinTrampolineSetup(x float64) float64
func asinAsm(x float64) float64

const haveArchAsinh = true

func archAsinh(x float64) float64
func asinhTrampolineSetup(x float64) float64
func asinhAsm(x float64) float64

const haveArchErf = true

func archErf(x float64) float64
func erfTrampolineSetup(x float64) float64
func erfAsm(x float64) float64

const haveArchErfc = true

func archErfc(x float64) float64
func erfcTrampolineSetup(x float64) float64
func erfcAsm(x float64) float64

const haveArchAtan = true

func archAtan(x float64) float64
func atanTrampolineSetup(x float64) float64
func atanAsm(x float64) float64

const haveArchAtan2 = true

func archAtan2(y, x float64) float64
func atan2TrampolineSetup(x, y float64) float64
func atan2Asm(x, y float64) float64

const haveArchCbrt = true

func archCbrt(x float64) float64
func cbrtTrampolineSetup(x float64) float64
func cbrtAsm(x float64) float64

const haveArchTan = true

func archTan(x float64) float64
func tanTrampolineSetup(x float64) float64
func tanAsm(x float64) float64

const haveArchExpm1 = true

func archExpm1(x float64) float64
func expm1TrampolineSetup(x float64) float64
func expm1Asm(x float64) float64

const haveArchPow = true

func archPow(x, y float64) float64
func powTrampolineSetup(x, y float64) float64
func powAsm(x, y float64) float64

const haveArchFrexp = false

func archFrexp(x float64) (float64, int) {
	panic("not implemented")
}

const haveArchLdexp = false

func archLdexp(frac float64, exp int) float64 {
	panic("not implemented")
}

const haveArchLog2 = false

func archLog2(x float64) float64 {
	panic("not implemented")
}

const haveArchMod = false

func archMod(x, y float64) float64 {
	panic("not implemented")
}

const haveArchRemainder = false

func archRemainder(x, y float64) float64 {
	panic("not implemented")
}

// hasVX reports whether the machine has the z/Architecture
// vector facility installed and enabled.
var hasVX = cpu.S390X.HasVX
