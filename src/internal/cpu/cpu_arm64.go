// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cpu

// CacheLinePadSize is used to prevent false sharing of cache lines.
// We choose 128 because Apple Silicon, a.k.a. M1, has 128-byte cache line size.
// It doesn't cost much and is much more future-proof.
const CacheLinePadSize = 128

func doinit() {
	options = []option{
		{Name: "aes", Feature: &ARM64.HasAES},
		{Name: "pmull", Feature: &ARM64.HasPMULL},
		{Name: "sha1", Feature: &ARM64.HasSHA1},
		{Name: "sha2", Feature: &ARM64.HasSHA2},
		{Name: "sha512", Feature: &ARM64.HasSHA512},
		{Name: "sha3", Feature: &ARM64.HasSHA3},
		{Name: "crc32", Feature: &ARM64.HasCRC32},
		{Name: "atomics", Feature: &ARM64.HasATOMICS},
		{Name: "cpuid", Feature: &ARM64.HasCPUID},
		{Name: "isNeoverse", Feature: &ARM64.IsNeoverse},
	}

	// arm64 uses different ways to detect CPU features at runtime depending on the operating system.
	osInit()
}

func getisar0() uint64

func getpfr0() uint64

func getMIDR() uint64

func extractBits(data uint64, start, end uint) uint {
	return (uint)(data>>start) & ((1 << (end - start + 1)) - 1)
}

func parseARM64SystemRegisters(isar0, pfr0 uint64) {
	// ID_AA64ISAR0_EL1
	// https://developer.arm.com/documentation/ddi0601/2025-03/AArch64-Registers/ID-AA64ISAR0-EL1--AArch64-Instruction-Set-Attribute-Register-0
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

	switch extractBits(isar0, 32, 35) {
	case 1:
		ARM64.HasSHA3 = true
	}

	switch extractBits(pfr0, 48, 51) {
	case 1:
		ARM64.HasDIT = true
	}
}
