// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sys

import "encoding/binary"

// ArchFamily represents a family of one or more related architectures.
// For example, amd64 and amd64p32 are both members of the AMD64 family,
// and ppc64 and ppc64le are both members of the PPC64 family.
type ArchFamily byte

const (
	AMD64 ArchFamily = iota
	ARM
	ARM64
	I386
	MIPS
	MIPS64
	PPC64
	S390X
)

// Arch represents an individual architecture.
type Arch struct {
	Name   string
	Family ArchFamily

	ByteOrder binary.ByteOrder

	IntSize int
	PtrSize int
	RegSize int

	// MinLC is the minimum length of an instruction code.
	MinLC int
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
	Name:      "386",
	Family:    I386,
	ByteOrder: binary.LittleEndian,
	IntSize:   4,
	PtrSize:   4,
	RegSize:   4,
	MinLC:     1,
}

var ArchAMD64 = &Arch{
	Name:      "amd64",
	Family:    AMD64,
	ByteOrder: binary.LittleEndian,
	IntSize:   8,
	PtrSize:   8,
	RegSize:   8,
	MinLC:     1,
}

var ArchAMD64P32 = &Arch{
	Name:      "amd64p32",
	Family:    AMD64,
	ByteOrder: binary.LittleEndian,
	IntSize:   4,
	PtrSize:   4,
	RegSize:   8,
	MinLC:     1,
}

var ArchARM = &Arch{
	Name:      "arm",
	Family:    ARM,
	ByteOrder: binary.LittleEndian,
	IntSize:   4,
	PtrSize:   4,
	RegSize:   4,
	MinLC:     4,
}

var ArchARM64 = &Arch{
	Name:      "arm64",
	Family:    ARM64,
	ByteOrder: binary.LittleEndian,
	IntSize:   8,
	PtrSize:   8,
	RegSize:   8,
	MinLC:     4,
}

var ArchMIPS = &Arch{
	Name:      "mips",
	Family:    MIPS,
	ByteOrder: binary.BigEndian,
	IntSize:   4,
	PtrSize:   4,
	RegSize:   4,
	MinLC:     4,
}

var ArchMIPSLE = &Arch{
	Name:      "mipsle",
	Family:    MIPS,
	ByteOrder: binary.LittleEndian,
	IntSize:   4,
	PtrSize:   4,
	RegSize:   4,
	MinLC:     4,
}

var ArchMIPS64 = &Arch{
	Name:      "mips64",
	Family:    MIPS64,
	ByteOrder: binary.BigEndian,
	IntSize:   8,
	PtrSize:   8,
	RegSize:   8,
	MinLC:     4,
}

var ArchMIPS64LE = &Arch{
	Name:      "mips64le",
	Family:    MIPS64,
	ByteOrder: binary.LittleEndian,
	IntSize:   8,
	PtrSize:   8,
	RegSize:   8,
	MinLC:     4,
}

var ArchPPC64 = &Arch{
	Name:      "ppc64",
	Family:    PPC64,
	ByteOrder: binary.BigEndian,
	IntSize:   8,
	PtrSize:   8,
	RegSize:   8,
	MinLC:     4,
}

var ArchPPC64LE = &Arch{
	Name:      "ppc64le",
	Family:    PPC64,
	ByteOrder: binary.LittleEndian,
	IntSize:   8,
	PtrSize:   8,
	RegSize:   8,
	MinLC:     4,
}

var ArchS390X = &Arch{
	Name:      "s390x",
	Family:    S390X,
	ByteOrder: binary.BigEndian,
	IntSize:   8,
	PtrSize:   8,
	RegSize:   8,
	MinLC:     2,
}

var Archs = [...]*Arch{
	Arch386,
	ArchAMD64,
	ArchAMD64P32,
	ArchARM,
	ArchARM64,
	ArchMIPS,
	ArchMIPSLE,
	ArchMIPS64,
	ArchMIPS64LE,
	ArchPPC64,
	ArchPPC64LE,
	ArchS390X,
}
