// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package cpu implements processor feature detection
// used by the Go standard library.
package cpu

var X86 x86

// The booleans in x86 contain the correspondingly named cpuid feature bit.
// HasAVX and HasAVX2 are only set if the OS does support XMM and YMM registers
// in addition to the cpuid feature bit being set.
// The struct is padded to avoid false sharing.
type x86 struct {
	_            [CacheLineSize]byte
	HasAES       bool
	HasADX       bool
	HasAVX       bool
	HasAVX2      bool
	HasBMI1      bool
	HasBMI2      bool
	HasERMS      bool
	HasFMA       bool
	HasOSXSAVE   bool
	HasPCLMULQDQ bool
	HasPOPCNT    bool
	HasSSE2      bool
	HasSSE3      bool
	HasSSSE3     bool
	HasSSE41     bool
	HasSSE42     bool
	_            [CacheLineSize]byte
}

var PPC64 ppc64

// For ppc64x, it is safe to check only for ISA level starting on ISA v3.00,
// since there are no optional categories. There are some exceptions that also
// require kernel support to work (darn, scv), so there are capability bits for
// those as well. The minimum processor requirement is POWER8 (ISA 2.07), so we
// maintain some of the old capability checks for optional categories for
// safety.
// The struct is padded to avoid false sharing.
type ppc64 struct {
	_          [CacheLineSize]byte
	HasVMX     bool // Vector unit (Altivec)
	HasDFP     bool // Decimal Floating Point unit
	HasVSX     bool // Vector-scalar unit
	HasHTM     bool // Hardware Transactional Memory
	HasISEL    bool // Integer select
	HasVCRYPTO bool // Vector cryptography
	HasHTMNOSC bool // HTM: kernel-aborted transaction in syscalls
	HasDARN    bool // Hardware random number generator (requires kernel enablement)
	HasSCV     bool // Syscall vectored (requires kernel enablement)
	IsPOWER8   bool // ISA v2.07 (POWER8)
	IsPOWER9   bool // ISA v3.00 (POWER9)
	_          [CacheLineSize]byte
}

var ARM64 arm64

// The booleans in arm64 contain the correspondingly named cpu feature bit.
// The struct is padded to avoid false sharing.
type arm64 struct {
	_          [CacheLineSize]byte
	HasFP      bool
	HasASIMD   bool
	HasEVTSTRM bool
	HasAES     bool
	HasPMULL   bool
	HasSHA1    bool
	HasSHA2    bool
	HasCRC32   bool
	HasATOMICS bool
	_          [CacheLineSize]byte
}
