// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "internal/cpu"

const (
	hwcap_FP       = 1 << 0
	hwcap_ASIMD    = 1 << 1
	hwcap_EVTSTRM  = 1 << 2
	hwcap_AES      = 1 << 3
	hwcap_PMULL    = 1 << 4
	hwcap_SHA1     = 1 << 5
	hwcap_SHA2     = 1 << 6
	hwcap_CRC32    = 1 << 7
	hwcap_ATOMICS  = 1 << 8
	hwcap_FPHP     = 1 << 9
	hwcap_ASIMDHP  = 1 << 10
	hwcap_CPUID    = 1 << 11
	hwcap_ASIMDRDM = 1 << 12
	hwcap_JSCVT    = 1 << 13
	hwcap_FCMA     = 1 << 14
	hwcap_LRCPC    = 1 << 15
	hwcap_DCPOP    = 1 << 16
	hwcap_SHA3     = 1 << 17
	hwcap_SM3      = 1 << 18
	hwcap_SM4      = 1 << 19
	hwcap_ASIMDDP  = 1 << 20
	hwcap_SHA512   = 1 << 21
	hwcap_SVE      = 1 << 22
	hwcap_ASIMDFHM = 1 << 23
)

func getisar0() uint64
func getisar1() uint64
func getpfr0() uint64

// no hwcap support on FreeBSD aarch64, we need to retrieve the info from
// ID_AA64ISAR0_EL1, ID_AA64ISAR1_EL1 and ID_AA64PFR0_EL1
func archauxv(tag, val uintptr) {
	var isar0, isar1, pfr0 uint64

	isar0 = getisar0()
	isar1 = getisar1()
	pfr0 = getpfr0()

	// ID_AA64ISAR0_EL1
	switch extractBits(isar0, 4, 7) {
	case 1:
		cpu.HWCap |= hwcap_AES
	case 2:
		cpu.HWCap |= hwcap_PMULL | hwcap_AES
	}

	switch extractBits(isar0, 8, 11) {
	case 1:
		cpu.HWCap |= hwcap_SHA1
	}

	switch extractBits(isar0, 12, 15) {
	case 1:
		cpu.HWCap |= hwcap_SHA2
	case 2:
		cpu.HWCap |= hwcap_SHA2 | hwcap_SHA512
	}

	switch extractBits(isar0, 16, 19) {
	case 1:
		cpu.HWCap |= hwcap_CRC32
	}

	switch extractBits(isar0, 20, 23) {
	case 2:
		cpu.HWCap |= hwcap_ATOMICS
	}

	switch extractBits(isar0, 28, 31) {
	case 1:
		cpu.HWCap |= hwcap_ASIMDRDM
	}

	switch extractBits(isar0, 32, 35) {
	case 1:
		cpu.HWCap |= hwcap_SHA3
	}

	switch extractBits(isar0, 36, 39) {
	case 1:
		cpu.HWCap |= hwcap_SM3
	}

	switch extractBits(isar0, 40, 43) {
	case 1:
		cpu.HWCap |= hwcap_SM4
	}

	switch extractBits(isar0, 44, 47) {
	case 1:
		cpu.HWCap |= hwcap_ASIMDDP
	}

	// ID_AA64ISAR1_EL1
	switch extractBits(isar1, 0, 3) {
	case 1:
		cpu.HWCap |= hwcap_DCPOP
	}

	switch extractBits(isar1, 12, 15) {
	case 1:
		cpu.HWCap |= hwcap_JSCVT
	}

	switch extractBits(isar1, 16, 19) {
	case 1:
		cpu.HWCap |= hwcap_FCMA
	}

	switch extractBits(isar1, 20, 23) {
	case 1:
		cpu.HWCap |= hwcap_LRCPC
	}

	// ID_AA64PFR0_EL1
	switch extractBits(pfr0, 16, 19) {
	case 0:
		cpu.HWCap |= hwcap_FP
	case 1:
		cpu.HWCap |= hwcap_FP | hwcap_FPHP
	}

	switch extractBits(pfr0, 20, 23) {
	case 0:
		cpu.HWCap |= hwcap_ASIMD
	case 1:
		cpu.HWCap |= hwcap_ASIMD | hwcap_ASIMDHP
	}

	switch extractBits(pfr0, 32, 35) {
	case 1:
		cpu.HWCap |= hwcap_SVE
	}
}

func extractBits(data uint64, start, end uint) uint {
	return (uint)(data>>start) & ((1 << (end - start + 1)) - 1)
}

//go:nosplit
func cputicks() int64 {
	// Currently cputicks() is used in blocking profiler and to seed fastrand().
	// nanotime() is a poor approximation of CPU ticks that is enough for the profiler.
	return nanotime()
}
