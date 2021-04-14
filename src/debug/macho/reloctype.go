// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package macho

//go:generate stringer -type=RelocTypeGeneric,RelocTypeX86_64,RelocTypeARM,RelocTypeARM64 -output reloctype_string.go

type RelocTypeGeneric int

const (
	GENERIC_RELOC_VANILLA        RelocTypeGeneric = 0
	GENERIC_RELOC_PAIR           RelocTypeGeneric = 1
	GENERIC_RELOC_SECTDIFF       RelocTypeGeneric = 2
	GENERIC_RELOC_PB_LA_PTR      RelocTypeGeneric = 3
	GENERIC_RELOC_LOCAL_SECTDIFF RelocTypeGeneric = 4
	GENERIC_RELOC_TLV            RelocTypeGeneric = 5
)

func (r RelocTypeGeneric) GoString() string { return "macho." + r.String() }

type RelocTypeX86_64 int

const (
	X86_64_RELOC_UNSIGNED   RelocTypeX86_64 = 0
	X86_64_RELOC_SIGNED     RelocTypeX86_64 = 1
	X86_64_RELOC_BRANCH     RelocTypeX86_64 = 2
	X86_64_RELOC_GOT_LOAD   RelocTypeX86_64 = 3
	X86_64_RELOC_GOT        RelocTypeX86_64 = 4
	X86_64_RELOC_SUBTRACTOR RelocTypeX86_64 = 5
	X86_64_RELOC_SIGNED_1   RelocTypeX86_64 = 6
	X86_64_RELOC_SIGNED_2   RelocTypeX86_64 = 7
	X86_64_RELOC_SIGNED_4   RelocTypeX86_64 = 8
	X86_64_RELOC_TLV        RelocTypeX86_64 = 9
)

func (r RelocTypeX86_64) GoString() string { return "macho." + r.String() }

type RelocTypeARM int

const (
	ARM_RELOC_VANILLA        RelocTypeARM = 0
	ARM_RELOC_PAIR           RelocTypeARM = 1
	ARM_RELOC_SECTDIFF       RelocTypeARM = 2
	ARM_RELOC_LOCAL_SECTDIFF RelocTypeARM = 3
	ARM_RELOC_PB_LA_PTR      RelocTypeARM = 4
	ARM_RELOC_BR24           RelocTypeARM = 5
	ARM_THUMB_RELOC_BR22     RelocTypeARM = 6
	ARM_THUMB_32BIT_BRANCH   RelocTypeARM = 7
	ARM_RELOC_HALF           RelocTypeARM = 8
	ARM_RELOC_HALF_SECTDIFF  RelocTypeARM = 9
)

func (r RelocTypeARM) GoString() string { return "macho." + r.String() }

type RelocTypeARM64 int

const (
	ARM64_RELOC_UNSIGNED            RelocTypeARM64 = 0
	ARM64_RELOC_SUBTRACTOR          RelocTypeARM64 = 1
	ARM64_RELOC_BRANCH26            RelocTypeARM64 = 2
	ARM64_RELOC_PAGE21              RelocTypeARM64 = 3
	ARM64_RELOC_PAGEOFF12           RelocTypeARM64 = 4
	ARM64_RELOC_GOT_LOAD_PAGE21     RelocTypeARM64 = 5
	ARM64_RELOC_GOT_LOAD_PAGEOFF12  RelocTypeARM64 = 6
	ARM64_RELOC_POINTER_TO_GOT      RelocTypeARM64 = 7
	ARM64_RELOC_TLVP_LOAD_PAGE21    RelocTypeARM64 = 8
	ARM64_RELOC_TLVP_LOAD_PAGEOFF12 RelocTypeARM64 = 9
	ARM64_RELOC_ADDEND              RelocTypeARM64 = 10
)

func (r RelocTypeARM64) GoString() string { return "macho." + r.String() }
