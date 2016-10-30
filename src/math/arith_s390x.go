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

// hasVectorFacility reports whether the machine has the z/Architecture
// vector facility installed and enabled.
func hasVectorFacility() bool

var hasVX = hasVectorFacility()
