// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build 386 || amd64

package cpu

// DataCacheSizes returns the size of each data cache from lowest
// level in the hierarchy to highest.
//
// Unlike other parts of this package's public API, it is not safe
// to reference early in runtime initialization because it allocates.
// It's intended for testing only.
func DataCacheSizes() []uintptr {
	maxFunctionInformation, ebx0, ecx0, edx0 := cpuid(0, 0)
	if maxFunctionInformation < 1 {
		return nil
	}

	switch {
	// Check for "GenuineIntel"
	case ebx0 == 0x756E6547 && ecx0 == 0x6C65746E && edx0 == 0x49656E69:
		return getDataCacheSizesIntel(maxFunctionInformation)
	// Check for "AuthenticAMD"
	case ebx0 == 0x68747541 && ecx0 == 0x444D4163 && edx0 == 0x69746E65:
		return getDataCacheSizesAMD()
	}
	return nil
}

func extractBits(arg uint32, l int, r int) uint32 {
	if l > r {
		panic("bad bit range")
	}
	return (arg >> l) & ((1 << (r - l + 1)) - 1)
}

func getDataCacheSizesIntel(maxID uint32) []uintptr {
	// Constants for cache types
	const (
		noCache          = 0
		dataCache        = 1
		instructionCache = 2
		unifiedCache     = 3
	)
	if maxID < 4 {
		return nil
	}

	// Iterate through CPUID leaf 4 (deterministic cache parameters)
	var caches []uintptr
	for i := uint32(0); i < 0xFFFF; i++ {
		eax, ebx, ecx, _ := cpuid(4, i)

		cacheType := eax & 0xF // EAX bits 4-0: Cache Type
		if cacheType == 0 {
			break
		}

		// Report only data caches.
		if !(cacheType == dataCache || cacheType == unifiedCache) {
			continue
		}

		// Guaranteed to always start counting from 1.
		level := (eax >> 5) & 0x7

		lineSize := extractBits(ebx, 0, 11) + 1         // Bits 11-0: Line size in bytes - 1
		partitions := extractBits(ebx, 12, 21) + 1      // Bits 21-12: Physical line partitions - 1
		ways := extractBits(ebx, 22, 31) + 1            // Bits 31-22: Ways of associativity - 1
		sets := uint64(ecx) + 1                         // Number of sets - 1
		size := uint64(ways*partitions*lineSize) * sets // Calculate cache size in bytes

		caches = append(caches, uintptr(size))

		// If we see more than one cache described per level, or they appear
		// out of order, crash.
		//
		// Going by the SDM, it's not clear whether this is actually possible,
		// so this code is purely defensive.
		if level != uint32(len(caches)) {
			panic("expected levels to be in order and for there to be one data/unified cache per level")
		}
	}
	return caches
}

func getDataCacheSizesAMD() []uintptr {
	maxExtendedFunctionInformation, _, _, _ := cpuid(0x80000000, 0)
	if maxExtendedFunctionInformation < 0x80000006 {
		return nil
	}

	var caches []uintptr

	_, _, ecx5, _ := cpuid(0x80000005, 0)
	_, _, ecx6, edx6 := cpuid(0x80000006, 0)

	// The size is return in kb, turning into bytes.
	l1dSize := uintptr(extractBits(ecx5, 24, 31) << 10)
	caches = append(caches, l1dSize)

	// Check that L2 cache is present.
	if l2Assoc := extractBits(ecx6, 12, 15); l2Assoc == 0 {
		return caches
	}
	l2Size := uintptr(extractBits(ecx6, 16, 31) << 10)
	caches = append(caches, l2Size)

	// Check that L3 cache is present.
	if l3Assoc := extractBits(edx6, 12, 15); l3Assoc == 0 {
		return caches
	}
	// Specifies the L3 cache size is within the following range:
	// (L3Size[31:18] * 512KB) <= L3 cache size < ((L3Size[31:18]+1) * 512KB).
	l3Size := uintptr(extractBits(edx6, 18, 31) * (512 << 10))
	caches = append(caches, l3Size)

	return caches
}
