// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build loong64

package cpu

// CacheLinePadSize is used to prevent false sharing of cache lines.
// We choose 64 because Loongson 3A5000 the L1 Dcache is 4-way 256-line 64-byte-per-line.
const CacheLinePadSize = 64

// Bit fields for CPUCFG registers, Related reference documents:
// https://loongson.github.io/LoongArch-Documentation/LoongArch-Vol1-EN.html#_cpucfg
const (
	// CPUCFG1 bits
	cpucfg1_CRC32 = 1 << 25

	// CPUCFG2 bits
	cpucfg2_LAM_BH = 1 << 27
	cpucfg2_LAMCAS = 1 << 28
)

// get_cpucfg is implemented in cpu_loong64.s.
func get_cpucfg(reg uint32) uint32

func doinit() {
	options = []option{
		{Name: "lsx", Feature: &Loong64.HasLSX},
		{Name: "lasx", Feature: &Loong64.HasLASX},
		{Name: "crc32", Feature: &Loong64.HasCRC32},
		{Name: "lamcas", Feature: &Loong64.HasLAMCAS},
		{Name: "lam_bh", Feature: &Loong64.HasLAM_BH},
	}

	// The CPUCFG data on Loong64 only reflects the hardware capabilities,
	// not the kernel support status, so features such as LSX and LASX that
	// require kernel support cannot be obtained from the CPUCFG data.
	//
	// These features only require hardware capability support and do not
	// require kernel specific support, so they can be obtained directly
	// through CPUCFG
	cfg1 := get_cpucfg(1)
	cfg2 := get_cpucfg(2)

	Loong64.HasCRC32 = cfgIsSet(cfg1, cpucfg1_CRC32)
	Loong64.HasLAMCAS = cfgIsSet(cfg2, cpucfg2_LAMCAS)
	Loong64.HasLAM_BH = cfgIsSet(cfg2, cpucfg2_LAM_BH)

	osInit()
}

func cfgIsSet(cfg uint32, val uint32) bool {
	return cfg&val != 0
}
