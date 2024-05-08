// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

import "runtime"

// cacheLineSize is used to prevent false sharing of cache lines.
// We choose 128 because Apple Silicon, a.k.a. M1, has 128-byte cache line size.
// It doesn't cost much and is much more future-proof.
const cacheLineSize = 128

func initOptions() {
	options = []option{
		{Name: "fp", Feature: &ARM64.HasFP},
		{Name: "asimd", Feature: &ARM64.HasASIMD},
		{Name: "evstrm", Feature: &ARM64.HasEVTSTRM},
		{Name: "aes", Feature: &ARM64.HasAES},
		{Name: "fphp", Feature: &ARM64.HasFPHP},
		{Name: "jscvt", Feature: &ARM64.HasJSCVT},
		{Name: "lrcpc", Feature: &ARM64.HasLRCPC},
		{Name: "pmull", Feature: &ARM64.HasPMULL},
		{Name: "sha1", Feature: &ARM64.HasSHA1},
		{Name: "sha2", Feature: &ARM64.HasSHA2},
		{Name: "sha3", Feature: &ARM64.HasSHA3},
		{Name: "sha512", Feature: &ARM64.HasSHA512},
		{Name: "sm3", Feature: &ARM64.HasSM3},
		{Name: "sm4", Feature: &ARM64.HasSM4},
		{Name: "sve", Feature: &ARM64.HasSVE},
		{Name: "sve2", Feature: &ARM64.HasSVE2},
		{Name: "crc32", Feature: &ARM64.HasCRC32},
		{Name: "atomics", Feature: &ARM64.HasATOMICS},
		{Name: "asimdhp", Feature: &ARM64.HasASIMDHP},
		{Name: "cpuid", Feature: &ARM64.HasCPUID},
		{Name: "asimrdm", Feature: &ARM64.HasASIMDRDM},
		{Name: "fcma", Feature: &ARM64.HasFCMA},
		{Name: "dcpop", Feature: &ARM64.HasDCPOP},
		{Name: "asimddp", Feature: &ARM64.HasASIMDDP},
		{Name: "asimdfhm", Feature: &ARM64.HasASIMDFHM},
	}
}

func archInit() {
	switch runtime.GOOS {
	case "freebsd":
		readARM64Registers()
	case "linux", "netbsd", "openbsd":
		doinit()
	default:
		// Many platforms don't seem to allow reading these registers.
		setMinimalFeatures()
	}
}

// setMinimalFeatures fakes the minimal ARM64 features expected by
// TestARM64minimalFeatures.
func setMinimalFeatures() {
	ARM64.HasASIMD = true
	ARM64.HasFP = true
}

func readARM64Registers() {
	Initialized = true

	parseARM64SystemRegisters(getisar0(), getisar1(), getpfr0())
}

func parseARM64SystemRegisters(isar0, isar1, pfr0 uint64) {
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
	case 1:
		ARM64.HasSHA2 = true
	case 2:
		ARM64.HasSHA2 = true
		ARM64.HasSHA512 = true
	}

	switch extractBits(isar0, 16, 19) {
	case 1:
		ARM64.HasCRC32 = true
	}

	switch extractBits(isar0, 20, 23) {
	case 2:
		ARM64.HasATOMICS = true
	}

	switch extractBits(isar0, 28, 31) {
	case 1:
		ARM64.HasASIMDRDM = true
	}

	switch extractBits(isar0, 32, 35) {
	case 1:
		ARM64.HasSHA3 = true
	}

	switch extractBits(isar0, 36, 39) {
	case 1:
		ARM64.HasSM3 = true
	}

	switch extractBits(isar0, 40, 43) {
	case 1:
		ARM64.HasSM4 = true
	}

	switch extractBits(isar0, 44, 47) {
	case 1:
		ARM64.HasASIMDDP = true
	}

	// ID_AA64ISAR1_EL1
	switch extractBits(isar1, 0, 3) {
	case 1:
		ARM64.HasDCPOP = true
	}

	switch extractBits(isar1, 12, 15) {
	case 1:
		ARM64.HasJSCVT = true
	}

	switch extractBits(isar1, 16, 19) {
	case 1:
		ARM64.HasFCMA = true
	}

	switch extractBits(isar1, 20, 23) {
	case 1:
		ARM64.HasLRCPC = true
	}

	// ID_AA64PFR0_EL1
	switch extractBits(pfr0, 16, 19) {
	case 0:
		ARM64.HasFP = true
	case 1:
		ARM64.HasFP = true
		ARM64.HasFPHP = true
	}

	switch extractBits(pfr0, 20, 23) {
	case 0:
		ARM64.HasASIMD = true
	case 1:
		ARM64.HasASIMD = true
		ARM64.HasASIMDHP = true
	}

	switch extractBits(pfr0, 32, 35) {
	case 1:
		ARM64.HasSVE = true

		parseARM64SVERegister(getzfr0())
	}
}

func parseARM64SVERegister(zfr0 uint64) {
	switch extractBits(zfr0, 0, 3) {
	case 1:
		ARM64.HasSVE2 = true
	}
}

func extractBits(data uint64, start, end uint) uint {
	return (uint)(data>>start) & ((1 << (end - start + 1)) - 1)
}
