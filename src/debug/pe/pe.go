// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package pe

type FileHeader struct {
	Machine              uint16
	NumberOfSections     uint16
	TimeDateStamp        uint32
	PointerToSymbolTable uint32
	NumberOfSymbols      uint32
	SizeOfOptionalHeader uint16
	Characteristics      uint16
}

type DataDirectory struct {
	VirtualAddress uint32
	Size           uint32
}

type OptionalHeader32 struct {
	Magic                       uint16
	MajorLinkerVersion          uint8
	MinorLinkerVersion          uint8
	SizeOfCode                  uint32
	SizeOfInitializedData       uint32
	SizeOfUninitializedData     uint32
	AddressOfEntryPoint         uint32
	BaseOfCode                  uint32
	BaseOfData                  uint32
	ImageBase                   uint32
	SectionAlignment            uint32
	FileAlignment               uint32
	MajorOperatingSystemVersion uint16
	MinorOperatingSystemVersion uint16
	MajorImageVersion           uint16
	MinorImageVersion           uint16
	MajorSubsystemVersion       uint16
	MinorSubsystemVersion       uint16
	Win32VersionValue           uint32
	SizeOfImage                 uint32
	SizeOfHeaders               uint32
	CheckSum                    uint32
	Subsystem                   uint16
	DllCharacteristics          uint16
	SizeOfStackReserve          uint32
	SizeOfStackCommit           uint32
	SizeOfHeapReserve           uint32
	SizeOfHeapCommit            uint32
	LoaderFlags                 uint32
	NumberOfRvaAndSizes         uint32
	DataDirectory               [16]DataDirectory
}

type OptionalHeader64 struct {
	Magic                       uint16
	MajorLinkerVersion          uint8
	MinorLinkerVersion          uint8
	SizeOfCode                  uint32
	SizeOfInitializedData       uint32
	SizeOfUninitializedData     uint32
	AddressOfEntryPoint         uint32
	BaseOfCode                  uint32
	ImageBase                   uint64
	SectionAlignment            uint32
	FileAlignment               uint32
	MajorOperatingSystemVersion uint16
	MinorOperatingSystemVersion uint16
	MajorImageVersion           uint16
	MinorImageVersion           uint16
	MajorSubsystemVersion       uint16
	MinorSubsystemVersion       uint16
	Win32VersionValue           uint32
	SizeOfImage                 uint32
	SizeOfHeaders               uint32
	CheckSum                    uint32
	Subsystem                   uint16
	DllCharacteristics          uint16
	SizeOfStackReserve          uint64
	SizeOfStackCommit           uint64
	SizeOfHeapReserve           uint64
	SizeOfHeapCommit            uint64
	LoaderFlags                 uint32
	NumberOfRvaAndSizes         uint32
	DataDirectory               [16]DataDirectory
}

const (
	IMAGE_FILE_MACHINE_UNKNOWN   = 0x0
	IMAGE_FILE_MACHINE_AM33      = 0x1d3
	IMAGE_FILE_MACHINE_AMD64     = 0x8664
	IMAGE_FILE_MACHINE_ARM       = 0x1c0
	IMAGE_FILE_MACHINE_EBC       = 0xebc
	IMAGE_FILE_MACHINE_I386      = 0x14c
	IMAGE_FILE_MACHINE_IA64      = 0x200
	IMAGE_FILE_MACHINE_M32R      = 0x9041
	IMAGE_FILE_MACHINE_MIPS16    = 0x266
	IMAGE_FILE_MACHINE_MIPSFPU   = 0x366
	IMAGE_FILE_MACHINE_MIPSFPU16 = 0x466
	IMAGE_FILE_MACHINE_POWERPC   = 0x1f0
	IMAGE_FILE_MACHINE_POWERPCFP = 0x1f1
	IMAGE_FILE_MACHINE_R4000     = 0x166
	IMAGE_FILE_MACHINE_SH3       = 0x1a2
	IMAGE_FILE_MACHINE_SH3DSP    = 0x1a3
	IMAGE_FILE_MACHINE_SH4       = 0x1a6
	IMAGE_FILE_MACHINE_SH5       = 0x1a8
	IMAGE_FILE_MACHINE_THUMB     = 0x1c2
	IMAGE_FILE_MACHINE_WCEMIPSV2 = 0x169
)
