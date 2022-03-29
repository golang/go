// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sys

import "encoding/binary"

// ArchFamily represents a family of one or more related architectures.
// For example, ppc64 and ppc64le are both members of the PPC64 family.
type ArchFamily byte

const (
	NoArch ArchFamily = iota
	AMD64
	ARM
	ARM64
	I386
	Loong64
	MIPS
	MIPS64
	PPC64
	RISCV64
	S390X
	Wasm
)

// Arch represents an individual architecture.
type Arch struct {
	Name   string
	Family ArchFamily

	ByteOrder binary.ByteOrder

	// PtrSize is the size in bytes of pointers and the
	// predeclared "int", "uint", and "uintptr" types.
	PtrSize int

	// RegSize is the size in bytes of general purpose registers.
	RegSize int

	// MinLC is the minimum length of an instruction code.
	MinLC int

	// Alignment is maximum alignment required by the architecture
	// for any (compiler-generated) load or store instruction.
	// Loads or stores smaller than Alignment must be naturally aligned.
	// Loads or stores larger than Alignment need only be Alignment-aligned.
	Alignment int8

	// CanMergeLoads reports whether the backend optimization passes
	// can combine adjacent loads into a single larger, possibly unaligned, load.
	// Note that currently the optimizations must be able to handle little endian byte order.
	CanMergeLoads bool
}

// InFamily reports whether a is a member of any of the specified
// architecture families.
func (a *Arch) InFamily(xs ...ArchFamily) bool {
	for _, x := range xs {
		if a.Family == x {
			return true
		}
	}
	return false
}

var Arch386 = &Arch{
	Name:          "386",
	Family:        I386,
	ByteOrder:     binary.LittleEndian,
	PtrSize:       4,
	RegSize:       4,
	MinLC:         1,
	Alignment:     1,
	CanMergeLoads: true,
}

var ArchAMD64 = &Arch{
	Name:          "amd64",
	Family:        AMD64,
	ByteOrder:     binary.LittleEndian,
	PtrSize:       8,
	RegSize:       8,
	MinLC:         1,
	Alignment:     1,
	CanMergeLoads: true,
}

var ArchARM = &Arch{
	Name:          "arm",
	Family:        ARM,
	ByteOrder:     binary.LittleEndian,
	PtrSize:       4,
	RegSize:       4,
	MinLC:         4,
	Alignment:     4, // TODO: just for arm5?
	CanMergeLoads: false,
}

var ArchARM64 = &Arch{
	Name:          "arm64",
	Family:        ARM64,
	ByteOrder:     binary.LittleEndian,
	PtrSize:       8,
	RegSize:       8,
	MinLC:         4,
	Alignment:     1,
	CanMergeLoads: true,
}

var ArchLoong64 = &Arch{
	Name:          "loong64",
	Family:        Loong64,
	ByteOrder:     binary.LittleEndian,
	PtrSize:       8,
	RegSize:       8,
	MinLC:         4,
	Alignment:     8, // Unaligned accesses are not guaranteed to be fast
	CanMergeLoads: false,
}

var ArchMIPS = &Arch{
	Name:          "mips",
	Family:        MIPS,
	ByteOrder:     binary.BigEndian,
	PtrSize:       4,
	RegSize:       4,
	MinLC:         4,
	Alignment:     4,
	CanMergeLoads: false,
}

var ArchMIPSLE = &Arch{
	Name:          "mipsle",
	Family:        MIPS,
	ByteOrder:     binary.LittleEndian,
	PtrSize:       4,
	RegSize:       4,
	MinLC:         4,
	Alignment:     4,
	CanMergeLoads: false,
}

var ArchMIPS64 = &Arch{
	Name:          "mips64",
	Family:        MIPS64,
	ByteOrder:     binary.BigEndian,
	PtrSize:       8,
	RegSize:       8,
	MinLC:         4,
	Alignment:     8,
	CanMergeLoads: false,
}

var ArchMIPS64LE = &Arch{
	Name:          "mips64le",
	Family:        MIPS64,
	ByteOrder:     binary.LittleEndian,
	PtrSize:       8,
	RegSize:       8,
	MinLC:         4,
	Alignment:     8,
	CanMergeLoads: false,
}

var ArchPPC64 = &Arch{
	Name:          "ppc64",
	Family:        PPC64,
	ByteOrder:     binary.BigEndian,
	PtrSize:       8,
	RegSize:       8,
	MinLC:         4,
	Alignment:     1,
	CanMergeLoads: false,
}

var ArchPPC64LE = &Arch{
	Name:          "ppc64le",
	Family:        PPC64,
	ByteOrder:     binary.LittleEndian,
	PtrSize:       8,
	RegSize:       8,
	MinLC:         4,
	Alignment:     1,
	CanMergeLoads: true,
}

var ArchRISCV64 = &Arch{
	Name:          "riscv64",
	Family:        RISCV64,
	ByteOrder:     binary.LittleEndian,
	PtrSize:       8,
	RegSize:       8,
	MinLC:         4,
	Alignment:     8, // riscv unaligned loads work, but are really slow (trap + simulated by OS)
	CanMergeLoads: false,
}

var ArchS390X = &Arch{
	Name:          "s390x",
	Family:        S390X,
	ByteOrder:     binary.BigEndian,
	PtrSize:       8,
	RegSize:       8,
	MinLC:         2,
	Alignment:     1,
	CanMergeLoads: true,
}

var ArchWasm = &Arch{
	Name:          "wasm",
	Family:        Wasm,
	ByteOrder:     binary.LittleEndian,
	PtrSize:       8,
	RegSize:       8,
	MinLC:         1,
	Alignment:     1,
	CanMergeLoads: false,
}

var Archs = [...]*Arch{
	Arch386,
	ArchAMD64,
	ArchARM,
	ArchARM64,
	ArchLoong64,
	ArchMIPS,
	ArchMIPSLE,
	ArchMIPS64,
	ArchMIPS64LE,
	ArchPPC64,
	ArchPPC64LE,
	ArchRISCV64,
	ArchS390X,
	ArchWasm,
}
