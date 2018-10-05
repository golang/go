// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Mach-O header data structures
// http://developer.apple.com/mac/library/documentation/DeveloperTools/Conceptual/MachORuntime/Reference/reference.html

package macho

import (
	"encoding/binary"
	"strconv"
)

// A FileHeader represents a Mach-O file header.
type FileHeader struct {
	Magic        uint32
	Cpu          Cpu
	SubCpu       uint32
	Type         HdrType
	NCommands    uint32 // number of load commands
	SizeCommands uint32 // size of all the load commands, not including this header.
	Flags        HdrFlags
}

func (h *FileHeader) Put(b []byte, o binary.ByteOrder) int {
	o.PutUint32(b[0:], h.Magic)
	o.PutUint32(b[4:], uint32(h.Cpu))
	o.PutUint32(b[8:], h.SubCpu)
	o.PutUint32(b[12:], uint32(h.Type))
	o.PutUint32(b[16:], h.NCommands)
	o.PutUint32(b[20:], h.SizeCommands)
	o.PutUint32(b[24:], uint32(h.Flags))
	if h.Magic == Magic32 {
		return 28
	}
	o.PutUint32(b[28:], 0)
	return 32
}

const (
	fileHeaderSize32 = 7 * 4
	fileHeaderSize64 = 8 * 4
)

const (
	Magic32  uint32 = 0xfeedface
	Magic64  uint32 = 0xfeedfacf
	MagicFat uint32 = 0xcafebabe
)

type HdrFlags uint32
type SegFlags uint32
type SecFlags uint32

// A HdrType is the Mach-O file type, e.g. an object file, executable, or dynamic library.
type HdrType uint32

const ( // SNAKE_CASE to CamelCase translation from C names
	MhObject  HdrType = 1
	MhExecute HdrType = 2
	MhCore    HdrType = 4
	MhDylib   HdrType = 6
	MhBundle  HdrType = 8
	MhDsym    HdrType = 0xa
)

var typeStrings = []intName{
	{uint32(MhObject), "Obj"},
	{uint32(MhExecute), "Exec"},
	{uint32(MhDylib), "Dylib"},
	{uint32(MhBundle), "Bundle"},
	{uint32(MhDsym), "Dsym"},
}

func (t HdrType) String() string   { return stringName(uint32(t), typeStrings, false) }
func (t HdrType) GoString() string { return stringName(uint32(t), typeStrings, true) }

// A Cpu is a Mach-O cpu type.
type Cpu uint32

const cpuArch64 = 0x01000000

const (
	Cpu386   Cpu = 7
	CpuAmd64 Cpu = Cpu386 | cpuArch64
	CpuArm   Cpu = 12
	CpuArm64 Cpu = CpuArm | cpuArch64
	CpuPpc   Cpu = 18
	CpuPpc64 Cpu = CpuPpc | cpuArch64
)

var cpuStrings = []intName{
	{uint32(Cpu386), "Cpu386"},
	{uint32(CpuAmd64), "CpuAmd64"},
	{uint32(CpuArm), "CpuArm"},
	{uint32(CpuArm64), "CpuArm64"},
	{uint32(CpuPpc), "CpuPpc"},
	{uint32(CpuPpc64), "CpuPpc64"},
}

func (i Cpu) String() string   { return stringName(uint32(i), cpuStrings, false) }
func (i Cpu) GoString() string { return stringName(uint32(i), cpuStrings, true) }

// A LoadCmd is a Mach-O load command.
type LoadCmd uint32

func (c LoadCmd) Command() LoadCmd { return c }

const ( // SNAKE_CASE to CamelCase translation from C names
	// Note 3 and 8 are obsolete
	LcSegment            LoadCmd = 0x1
	LcSymtab             LoadCmd = 0x2
	LcThread             LoadCmd = 0x4
	LcUnixthread         LoadCmd = 0x5 // thread+stack
	LcDysymtab           LoadCmd = 0xb
	LcDylib              LoadCmd = 0xc // load dylib command
	LcIdDylib            LoadCmd = 0xd // dynamically linked shared lib ident
	LcLoadDylinker       LoadCmd = 0xe // load a dynamic linker
	LcIdDylinker         LoadCmd = 0xf // id dylinker command (not load dylinker command)
	LcSegment64          LoadCmd = 0x19
	LcUuid               LoadCmd = 0x1b
	LcCodeSignature      LoadCmd = 0x1d
	LcSegmentSplitInfo   LoadCmd = 0x1e
	LcRpath              LoadCmd = 0x8000001c
	LcEncryptionInfo     LoadCmd = 0x21
	LcDyldInfo           LoadCmd = 0x22
	LcDyldInfoOnly       LoadCmd = 0x80000022
	LcVersionMinMacosx   LoadCmd = 0x24
	LcVersionMinIphoneos LoadCmd = 0x25
	LcFunctionStarts     LoadCmd = 0x26
	LcDyldEnvironment    LoadCmd = 0x27
	LcMain               LoadCmd = 0x80000028 // replacement for UnixThread
	LcDataInCode         LoadCmd = 0x29       // There are non-instructions in text
	LcSourceVersion      LoadCmd = 0x2a       // Source version used to build binary
	LcDylibCodeSignDrs   LoadCmd = 0x2b
	LcEncryptionInfo64   LoadCmd = 0x2c
	LcVersionMinTvos     LoadCmd = 0x2f
	LcVersionMinWatchos  LoadCmd = 0x30
)

var cmdStrings = []intName{
	{uint32(LcSegment), "LoadCmdSegment"},
	{uint32(LcThread), "LoadCmdThread"},
	{uint32(LcUnixthread), "LoadCmdUnixThread"},
	{uint32(LcDylib), "LoadCmdDylib"},
	{uint32(LcIdDylib), "LoadCmdIdDylib"},
	{uint32(LcLoadDylinker), "LoadCmdLoadDylinker"},
	{uint32(LcIdDylinker), "LoadCmdIdDylinker"},
	{uint32(LcSegment64), "LoadCmdSegment64"},
	{uint32(LcUuid), "LoadCmdUuid"},
	{uint32(LcRpath), "LoadCmdRpath"},
	{uint32(LcDyldEnvironment), "LoadCmdDyldEnv"},
	{uint32(LcMain), "LoadCmdMain"},
	{uint32(LcDataInCode), "LoadCmdDataInCode"},
	{uint32(LcSourceVersion), "LoadCmdSourceVersion"},
	{uint32(LcDyldInfo), "LoadCmdDyldInfo"},
	{uint32(LcDyldInfoOnly), "LoadCmdDyldInfoOnly"},
	{uint32(LcVersionMinMacosx), "LoadCmdMinOsx"},
	{uint32(LcFunctionStarts), "LoadCmdFunctionStarts"},
}

func (i LoadCmd) String() string   { return stringName(uint32(i), cmdStrings, false) }
func (i LoadCmd) GoString() string { return stringName(uint32(i), cmdStrings, true) }

type (
	// A Segment32 is a 32-bit Mach-O segment load command.
	Segment32 struct {
		LoadCmd
		Len     uint32
		Name    [16]byte
		Addr    uint32
		Memsz   uint32
		Offset  uint32
		Filesz  uint32
		Maxprot uint32
		Prot    uint32
		Nsect   uint32
		Flag    SegFlags
	}

	// A Segment64 is a 64-bit Mach-O segment load command.
	Segment64 struct {
		LoadCmd
		Len     uint32
		Name    [16]byte
		Addr    uint64
		Memsz   uint64
		Offset  uint64
		Filesz  uint64
		Maxprot uint32
		Prot    uint32
		Nsect   uint32
		Flag    SegFlags
	}

	// A SymtabCmd is a Mach-O symbol table command.
	SymtabCmd struct {
		LoadCmd
		Len     uint32
		Symoff  uint32
		Nsyms   uint32
		Stroff  uint32
		Strsize uint32
	}

	// A DysymtabCmd is a Mach-O dynamic symbol table command.
	DysymtabCmd struct {
		LoadCmd
		Len            uint32
		Ilocalsym      uint32
		Nlocalsym      uint32
		Iextdefsym     uint32
		Nextdefsym     uint32
		Iundefsym      uint32
		Nundefsym      uint32
		Tocoffset      uint32
		Ntoc           uint32
		Modtaboff      uint32
		Nmodtab        uint32
		Extrefsymoff   uint32
		Nextrefsyms    uint32
		Indirectsymoff uint32
		Nindirectsyms  uint32
		Extreloff      uint32
		Nextrel        uint32
		Locreloff      uint32
		Nlocrel        uint32
	}

	// A DylibCmd is a Mach-O load dynamic library command.
	DylibCmd struct {
		LoadCmd
		Len            uint32
		Name           uint32
		Time           uint32
		CurrentVersion uint32
		CompatVersion  uint32
	}

	// A DylinkerCmd is a Mach-O load dynamic linker or environment command.
	DylinkerCmd struct {
		LoadCmd
		Len  uint32
		Name uint32
	}

	// A RpathCmd is a Mach-O rpath command.
	RpathCmd struct {
		LoadCmd
		Len  uint32
		Path uint32
	}

	// A Thread is a Mach-O thread state command.
	Thread struct {
		LoadCmd
		Len  uint32
		Type uint32
		Data []uint32
	}

	// LC_DYLD_INFO, LC_DYLD_INFO_ONLY
	DyldInfoCmd struct {
		LoadCmd
		Len                      uint32
		RebaseOff, RebaseLen     uint32 // file offset and length; data contains segment indices
		BindOff, BindLen         uint32 // file offset and length; data contains segment indices
		WeakBindOff, WeakBindLen uint32 // file offset and length
		LazyBindOff, LazyBindLen uint32 // file offset and length
		ExportOff, ExportLen     uint32 // file offset and length
	}

	// LC_CODE_SIGNATURE, LC_SEGMENT_SPLIT_INFO, LC_FUNCTION_STARTS, LC_DATA_IN_CODE, LC_DYLIB_CODE_SIGN_DRS
	LinkEditDataCmd struct {
		LoadCmd
		Len              uint32
		DataOff, DataLen uint32 // file offset and length
	}

	// LC_ENCRYPTION_INFO, LC_ENCRYPTION_INFO_64
	EncryptionInfoCmd struct {
		LoadCmd
		Len                uint32
		CryptOff, CryptLen uint32 // file offset and length
		CryptId            uint32
	}

	UuidCmd struct {
		LoadCmd
		Len uint32
		Id  [16]byte
	}

	// TODO Commands below not fully supported yet.

	EntryPointCmd struct {
		LoadCmd
		Len       uint32
		EntryOff  uint64 // file offset
		StackSize uint64 // if not zero, initial stack size
	}

	NoteCmd struct {
		LoadCmd
		Len            uint32
		Name           [16]byte
		Offset, Filesz uint64 // file offset and length
	}
)

const (
	FlagNoUndefs              HdrFlags = 0x1
	FlagIncrLink              HdrFlags = 0x2
	FlagDyldLink              HdrFlags = 0x4
	FlagBindAtLoad            HdrFlags = 0x8
	FlagPrebound              HdrFlags = 0x10
	FlagSplitSegs             HdrFlags = 0x20
	FlagLazyInit              HdrFlags = 0x40
	FlagTwoLevel              HdrFlags = 0x80
	FlagForceFlat             HdrFlags = 0x100
	FlagNoMultiDefs           HdrFlags = 0x200
	FlagNoFixPrebinding       HdrFlags = 0x400
	FlagPrebindable           HdrFlags = 0x800
	FlagAllModsBound          HdrFlags = 0x1000
	FlagSubsectionsViaSymbols HdrFlags = 0x2000
	FlagCanonical             HdrFlags = 0x4000
	FlagWeakDefines           HdrFlags = 0x8000
	FlagBindsToWeak           HdrFlags = 0x10000
	FlagAllowStackExecution   HdrFlags = 0x20000
	FlagRootSafe              HdrFlags = 0x40000
	FlagSetuidSafe            HdrFlags = 0x80000
	FlagNoReexportedDylibs    HdrFlags = 0x100000
	FlagPIE                   HdrFlags = 0x200000
	FlagDeadStrippableDylib   HdrFlags = 0x400000
	FlagHasTLVDescriptors     HdrFlags = 0x800000
	FlagNoHeapExecution       HdrFlags = 0x1000000
	FlagAppExtensionSafe      HdrFlags = 0x2000000
)

// A Section32 is a 32-bit Mach-O section header.
type Section32 struct {
	Name     [16]byte
	Seg      [16]byte
	Addr     uint32
	Size     uint32
	Offset   uint32
	Align    uint32
	Reloff   uint32
	Nreloc   uint32
	Flags    SecFlags
	Reserve1 uint32
	Reserve2 uint32
}

// A Section64 is a 64-bit Mach-O section header.
type Section64 struct {
	Name     [16]byte
	Seg      [16]byte
	Addr     uint64
	Size     uint64
	Offset   uint32
	Align    uint32
	Reloff   uint32
	Nreloc   uint32
	Flags    SecFlags
	Reserve1 uint32
	Reserve2 uint32
	Reserve3 uint32
}

// An Nlist32 is a Mach-O 32-bit symbol table entry.
type Nlist32 struct {
	Name  uint32
	Type  uint8
	Sect  uint8
	Desc  uint16
	Value uint32
}

// An Nlist64 is a Mach-O 64-bit symbol table entry.
type Nlist64 struct {
	Name  uint32
	Type  uint8
	Sect  uint8
	Desc  uint16
	Value uint64
}

func (n *Nlist64) Put64(b []byte, o binary.ByteOrder) uint32 {
	o.PutUint32(b[0:], n.Name)
	b[4] = byte(n.Type)
	b[5] = byte(n.Sect)
	o.PutUint16(b[6:], n.Desc)
	o.PutUint64(b[8:], n.Value)
	return 8 + 8
}

func (n *Nlist64) Put32(b []byte, o binary.ByteOrder) uint32 {
	o.PutUint32(b[0:], n.Name)
	b[4] = byte(n.Type)
	b[5] = byte(n.Sect)
	o.PutUint16(b[6:], n.Desc)
	o.PutUint32(b[8:], uint32(n.Value))
	return 8 + 4
}

// Regs386 is the Mach-O 386 register structure.
type Regs386 struct {
	AX    uint32
	BX    uint32
	CX    uint32
	DX    uint32
	DI    uint32
	SI    uint32
	BP    uint32
	SP    uint32
	SS    uint32
	FLAGS uint32
	IP    uint32
	CS    uint32
	DS    uint32
	ES    uint32
	FS    uint32
	GS    uint32
}

// RegsAMD64 is the Mach-O AMD64 register structure.
type RegsAMD64 struct {
	AX    uint64
	BX    uint64
	CX    uint64
	DX    uint64
	DI    uint64
	SI    uint64
	BP    uint64
	SP    uint64
	R8    uint64
	R9    uint64
	R10   uint64
	R11   uint64
	R12   uint64
	R13   uint64
	R14   uint64
	R15   uint64
	IP    uint64
	FLAGS uint64
	CS    uint64
	FS    uint64
	GS    uint64
}

type intName struct {
	i uint32
	s string
}

func stringName(i uint32, names []intName, goSyntax bool) string {
	for _, n := range names {
		if n.i == i {
			if goSyntax {
				return "macho." + n.s
			}
			return n.s
		}
	}
	return "0x" + strconv.FormatUint(uint64(i), 16)
}
