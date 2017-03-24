// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package math

func log10TrampolineSetup(x float64) float64
func log10Asm(x float64) float64

func cosTrampolineSetup(x float64) float64
func cosAsm(x float64) float64

func coshTrampolineSetup(x float64) float64
func coshAsm(x float64) float64

func sinTrampolineSetup(x float64) float64
func sinAsm(x float64) float64

func sinhTrampolineSetup(x float64) float64
func sinhAsm(x float64) float64

func tanhTrampolineSetup(x float64) float64
func tanhAsm(x float64) float64

func log1pTrampolineSetup(x float64) float64
func log1pAsm(x float64) float64

func atanhTrampolineSetup(x float64) float64
func atanhAsm(x float64) float64

func acosTrampolineSetup(x float64) float64
func acosAsm(x float64) float64

func acoshTrampolineSetup(x float64) float64
func acoshAsm(x float64) float64

func asinTrampolineSetup(x float64) float64
func asinAsm(x float64) float64

func asinhTrampolineSetup(x float64) float64
func asinhAsm(x float64) float64

func erfTrampolineSetup(x float64) float64
func erfAsm(x float64) float64

func erfcTrampolineSetup(x float64) float64
func erfcAsm(x float64) float64

func atanTrampolineSetup(x float64) float64
func atanAsm(x float64) float64

func atan2TrampolineSetup(x, y float64) float64
func atan2Asm(x, y float64) float64

func cbrtTrampolineSetup(x float64) float64
func cbrtAsm(x float64) float64

func logTrampolineSetup(x float64) float64
func logAsm(x float64) float64

func tanTrampolineSetup(x float64) float64
func tanAsm(x float64) float64

func expTrampolineSetup(x float64) float64
func expAsm(x float64) float64

func expm1TrampolineSetup(x float64) float64
func expm1Asm(x float64) float64

func powTrampolineSetup(x, y float64) float64
func powAsm(x, y float64) float64

// hasVectorFacility reports whether the machine has the z/Architecture
// vector facility installed and enabled.
func hasVectorFacility() bool

var hasVX = hasVectorFacility()
