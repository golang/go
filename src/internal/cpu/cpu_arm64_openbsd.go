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
)

func extractBits(data uint64, start, end uint) uint {
	return (uint)(data>>start) & ((1 << (end - start + 1)) - 1)
}

//go:noescape
func sysctlUint64(mib []uint32) (uint64, bool)

func osInit() {
	// Get ID_AA64ISAR0 from sysctl.
	isar0, ok := sysctlUint64([]uint32{_CTL_MACHDEP, _CPU_ID_AA64ISAR0})
	if !ok {
		return
	}

	// ID_AA64ISAR0_EL1
	switch extractBits(isar0, 4, 7) {
	case 1:
		ARM64.HasAES = true
	case 2:
		ARM64.HasAES = true
		ARM64.HasPMULL = true
	}

	switch extractBits(isar0, 8, 11) {
	case 1:
		ARM64.HasSHA1 = true
	}

	switch extractBits(isar0, 12, 15) {
	case 1, 2:
		ARM64.HasSHA2 = true
	}

	switch extractBits(isar0, 16, 19) {
	case 1:
		ARM64.HasCRC32 = true
	}

	switch extractBits(isar0, 20, 23) {
	case 2:
		ARM64.HasATOMICS = true
	}
}
