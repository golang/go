// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build arm64

package cpu

func osInit() {
	// Retrieve info from system register ID_AA64ISAR0_EL1.
	isar0 := getisar0()
	prf0 := getpfr0()

	parseARM64SystemRegisters(isar0, prf0)
}
