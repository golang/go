// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build arm64

package cpu

const (
	// From OpenBSD's sys/sysctl.h.
	_CTL_MACHDEP = 7

	// From OpenBSD's machine/cpu.h.
	_CPU_ID_AA64ISAR0 = 2
	_CPU_ID_AA64ISAR1 = 3
	_CPU_ID_AA64PFR0  = 8
)

//go:noescape
func sysctlUint64(mib []uint32) (uint64, bool)

func osInit() {
	// Get ID_AA64ISAR0 from sysctl.
	isar0, ok := sysctlUint64([]uint32{_CTL_MACHDEP, _CPU_ID_AA64ISAR0})
	if !ok {
		return
	}
	// Get ID_AA64PFR0 from sysctl.
	pfr0, ok := sysctlUint64([]uint32{_CTL_MACHDEP, _CPU_ID_AA64PFR0})
	if !ok {
		return
	}

	parseARM64SystemRegisters(isar0, pfr0)
}
