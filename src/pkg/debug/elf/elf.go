/*
 * ELF constants and data structures
 *
 * Derived from:
 * $FreeBSD: src/sys/sys/elf32.h,v 1.8.14.1 2005/12/30 22:13:58 marcel Exp $
 * $FreeBSD: src/sys/sys/elf64.h,v 1.10.14.1 2005/12/30 22:13:58 marcel Exp $
 * $FreeBSD: src/sys/sys/elf_common.h,v 1.15.8.1 2005/12/30 22:13:58 marcel Exp $
 * $FreeBSD: src/sys/alpha/include/elf.h,v 1.14 2003/09/25 01:10:22 peter Exp $
 * $FreeBSD: src/sys/amd64/include/elf.h,v 1.18 2004/08/03 08:21:48 dfr Exp $
 * $FreeBSD: src/sys/arm/include/elf.h,v 1.5.2.1 2006/06/30 21:42:52 cognet Exp $
 * $FreeBSD: src/sys/i386/include/elf.h,v 1.16 2004/08/02 19:12:17 dfr Exp $
 * $FreeBSD: src/sys/powerpc/include/elf.h,v 1.7 2004/11/02 09:47:01 ssouhlal Exp $
 * $FreeBSD: src/sys/sparc64/include/elf.h,v 1.12 2003/09/25 01:10:26 peter Exp $
 *
 * Copyright (c) 1996-1998 John D. Polstra.  All rights reserved.
 * Copyright (c) 2001 David E. O'Brien
 * Portions Copyright 2009 The Go Authors.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

package elf

import "strconv"

/*
 * Constants
 */

// Indexes into the Header.Ident array.
const (
	EI_CLASS      = 4  /* Class of machine. */
	EI_DATA       = 5  /* Data format. */
	EI_VERSION    = 6  /* ELF format version. */
	EI_OSABI      = 7  /* Operating system / ABI identification */
	EI_ABIVERSION = 8  /* ABI version */
	EI_PAD        = 9  /* Start of padding (per SVR4 ABI). */
	EI_NIDENT     = 16 /* Size of e_ident array. */
)

// Initial magic number for ELF files.
const ELFMAG = "\177ELF"

// Version is found in Header.Ident[EI_VERSION] and Header.Version.
type Version byte

const (
	EV_NONE    Version = 0
	EV_CURRENT Version = 1
)

var versionStrings = []intName{
	{0, "EV_NONE"},
	{1, "EV_CURRENT"},
}

func (i Version) String() string   { return stringName(uint32(i), versionStrings, false) }
func (i Version) GoString() string { return stringName(uint32(i), versionStrings, true) }

// Class is found in Header.Ident[EI_CLASS] and Header.Class.
type Class byte

const (
	ELFCLASSNONE Class = 0 /* Unknown class. */
	ELFCLASS32   Class = 1 /* 32-bit architecture. */
	ELFCLASS64   Class = 2 /* 64-bit architecture. */
)

var classStrings = []intName{
	{0, "ELFCLASSNONE"},
	{1, "ELFCLASS32"},
	{2, "ELFCLASS64"},
}

func (i Class) String() string   { return stringName(uint32(i), classStrings, false) }
func (i Class) GoString() string { return stringName(uint32(i), classStrings, true) }

// Data is found in Header.Ident[EI_DATA] and Header.Data.
type Data byte

const (
	ELFDATANONE Data = 0 /* Unknown data format. */
	ELFDATA2LSB Data = 1 /* 2's complement little-endian. */
	ELFDATA2MSB Data = 2 /* 2's complement big-endian. */
)

var dataStrings = []intName{
	{0, "ELFDATANONE"},
	{1, "ELFDATA2LSB"},
	{2, "ELFDATA2MSB"},
}

func (i Data) String() string   { return stringName(uint32(i), dataStrings, false) }
func (i Data) GoString() string { return stringName(uint32(i), dataStrings, true) }

// OSABI is found in Header.Ident[EI_OSABI] and Header.OSABI.
type OSABI byte

const (
	ELFOSABI_NONE       OSABI = 0   /* UNIX System V ABI */
	ELFOSABI_HPUX       OSABI = 1   /* HP-UX operating system */
	ELFOSABI_NETBSD     OSABI = 2   /* NetBSD */
	ELFOSABI_LINUX      OSABI = 3   /* GNU/Linux */
	ELFOSABI_HURD       OSABI = 4   /* GNU/Hurd */
	ELFOSABI_86OPEN     OSABI = 5   /* 86Open common IA32 ABI */
	ELFOSABI_SOLARIS    OSABI = 6   /* Solaris */
	ELFOSABI_AIX        OSABI = 7   /* AIX */
	ELFOSABI_IRIX       OSABI = 8   /* IRIX */
	ELFOSABI_FREEBSD    OSABI = 9   /* FreeBSD */
	ELFOSABI_TRU64      OSABI = 10  /* TRU64 UNIX */
	ELFOSABI_MODESTO    OSABI = 11  /* Novell Modesto */
	ELFOSABI_OPENBSD    OSABI = 12  /* OpenBSD */
	ELFOSABI_OPENVMS    OSABI = 13  /* Open VMS */
	ELFOSABI_NSK        OSABI = 14  /* HP Non-Stop Kernel */
	ELFOSABI_ARM        OSABI = 97  /* ARM */
	ELFOSABI_STANDALONE OSABI = 255 /* Standalone (embedded) application */
)

var osabiStrings = []intName{
	{0, "ELFOSABI_NONE"},
	{1, "ELFOSABI_HPUX"},
	{2, "ELFOSABI_NETBSD"},
	{3, "ELFOSABI_LINUX"},
	{4, "ELFOSABI_HURD"},
	{5, "ELFOSABI_86OPEN"},
	{6, "ELFOSABI_SOLARIS"},
	{7, "ELFOSABI_AIX"},
	{8, "ELFOSABI_IRIX"},
	{9, "ELFOSABI_FREEBSD"},
	{10, "ELFOSABI_TRU64"},
	{11, "ELFOSABI_MODESTO"},
	{12, "ELFOSABI_OPENBSD"},
	{13, "ELFOSABI_OPENVMS"},
	{14, "ELFOSABI_NSK"},
	{97, "ELFOSABI_ARM"},
	{255, "ELFOSABI_STANDALONE"},
}

func (i OSABI) String() string   { return stringName(uint32(i), osabiStrings, false) }
func (i OSABI) GoString() string { return stringName(uint32(i), osabiStrings, true) }

// Type is found in Header.Type.
type Type uint16

const (
	ET_NONE   Type = 0      /* Unknown type. */
	ET_REL    Type = 1      /* Relocatable. */
	ET_EXEC   Type = 2      /* Executable. */
	ET_DYN    Type = 3      /* Shared object. */
	ET_CORE   Type = 4      /* Core file. */
	ET_LOOS   Type = 0xfe00 /* First operating system specific. */
	ET_HIOS   Type = 0xfeff /* Last operating system-specific. */
	ET_LOPROC Type = 0xff00 /* First processor-specific. */
	ET_HIPROC Type = 0xffff /* Last processor-specific. */
)

var typeStrings = []intName{
	{0, "ET_NONE"},
	{1, "ET_REL"},
	{2, "ET_EXEC"},
	{3, "ET_DYN"},
	{4, "ET_CORE"},
	{0xfe00, "ET_LOOS"},
	{0xfeff, "ET_HIOS"},
	{0xff00, "ET_LOPROC"},
	{0xffff, "ET_HIPROC"},
}

func (i Type) String() string   { return stringName(uint32(i), typeStrings, false) }
func (i Type) GoString() string { return stringName(uint32(i), typeStrings, true) }

// Machine is found in Header.Machine.
type Machine uint16

const (
	EM_NONE        Machine = 0  /* Unknown machine. */
	EM_M32         Machine = 1  /* AT&T WE32100. */
	EM_SPARC       Machine = 2  /* Sun SPARC. */
	EM_386         Machine = 3  /* Intel i386. */
	EM_68K         Machine = 4  /* Motorola 68000. */
	EM_88K         Machine = 5  /* Motorola 88000. */
	EM_860         Machine = 7  /* Intel i860. */
	EM_MIPS        Machine = 8  /* MIPS R3000 Big-Endian only. */
	EM_S370        Machine = 9  /* IBM System/370. */
	EM_MIPS_RS3_LE Machine = 10 /* MIPS R3000 Little-Endian. */
	EM_PARISC      Machine = 15 /* HP PA-RISC. */
	EM_VPP500      Machine = 17 /* Fujitsu VPP500. */
	EM_SPARC32PLUS Machine = 18 /* SPARC v8plus. */
	EM_960         Machine = 19 /* Intel 80960. */
	EM_PPC         Machine = 20 /* PowerPC 32-bit. */
	EM_PPC64       Machine = 21 /* PowerPC 64-bit. */
	EM_S390        Machine = 22 /* IBM System/390. */
	EM_V800        Machine = 36 /* NEC V800. */
	EM_FR20        Machine = 37 /* Fujitsu FR20. */
	EM_RH32        Machine = 38 /* TRW RH-32. */
	EM_RCE         Machine = 39 /* Motorola RCE. */
	EM_ARM         Machine = 40 /* ARM. */
	EM_SH          Machine = 42 /* Hitachi SH. */
	EM_SPARCV9     Machine = 43 /* SPARC v9 64-bit. */
	EM_TRICORE     Machine = 44 /* Siemens TriCore embedded processor. */
	EM_ARC         Machine = 45 /* Argonaut RISC Core. */
	EM_H8_300      Machine = 46 /* Hitachi H8/300. */
	EM_H8_300H     Machine = 47 /* Hitachi H8/300H. */
	EM_H8S         Machine = 48 /* Hitachi H8S. */
	EM_H8_500      Machine = 49 /* Hitachi H8/500. */
	EM_IA_64       Machine = 50 /* Intel IA-64 Processor. */
	EM_MIPS_X      Machine = 51 /* Stanford MIPS-X. */
	EM_COLDFIRE    Machine = 52 /* Motorola ColdFire. */
	EM_68HC12      Machine = 53 /* Motorola M68HC12. */
	EM_MMA         Machine = 54 /* Fujitsu MMA. */
	EM_PCP         Machine = 55 /* Siemens PCP. */
	EM_NCPU        Machine = 56 /* Sony nCPU. */
	EM_NDR1        Machine = 57 /* Denso NDR1 microprocessor. */
	EM_STARCORE    Machine = 58 /* Motorola Star*Core processor. */
	EM_ME16        Machine = 59 /* Toyota ME16 processor. */
	EM_ST100       Machine = 60 /* STMicroelectronics ST100 processor. */
	EM_TINYJ       Machine = 61 /* Advanced Logic Corp. TinyJ processor. */
	EM_X86_64      Machine = 62 /* Advanced Micro Devices x86-64 */

	/* Non-standard or deprecated. */
	EM_486         Machine = 6      /* Intel i486. */
	EM_MIPS_RS4_BE Machine = 10     /* MIPS R4000 Big-Endian */
	EM_ALPHA_STD   Machine = 41     /* Digital Alpha (standard value). */
	EM_ALPHA       Machine = 0x9026 /* Alpha (written in the absence of an ABI) */
)

var machineStrings = []intName{
	{0, "EM_NONE"},
	{1, "EM_M32"},
	{2, "EM_SPARC"},
	{3, "EM_386"},
	{4, "EM_68K"},
	{5, "EM_88K"},
	{7, "EM_860"},
	{8, "EM_MIPS"},
	{9, "EM_S370"},
	{10, "EM_MIPS_RS3_LE"},
	{15, "EM_PARISC"},
	{17, "EM_VPP500"},
	{18, "EM_SPARC32PLUS"},
	{19, "EM_960"},
	{20, "EM_PPC"},
	{21, "EM_PPC64"},
	{22, "EM_S390"},
	{36, "EM_V800"},
	{37, "EM_FR20"},
	{38, "EM_RH32"},
	{39, "EM_RCE"},
	{40, "EM_ARM"},
	{42, "EM_SH"},
	{43, "EM_SPARCV9"},
	{44, "EM_TRICORE"},
	{45, "EM_ARC"},
	{46, "EM_H8_300"},
	{47, "EM_H8_300H"},
	{48, "EM_H8S"},
	{49, "EM_H8_500"},
	{50, "EM_IA_64"},
	{51, "EM_MIPS_X"},
	{52, "EM_COLDFIRE"},
	{53, "EM_68HC12"},
	{54, "EM_MMA"},
	{55, "EM_PCP"},
	{56, "EM_NCPU"},
	{57, "EM_NDR1"},
	{58, "EM_STARCORE"},
	{59, "EM_ME16"},
	{60, "EM_ST100"},
	{61, "EM_TINYJ"},
	{62, "EM_X86_64"},

	/* Non-standard or deprecated. */
	{6, "EM_486"},
	{10, "EM_MIPS_RS4_BE"},
	{41, "EM_ALPHA_STD"},
	{0x9026, "EM_ALPHA"},
}

func (i Machine) String() string   { return stringName(uint32(i), machineStrings, false) }
func (i Machine) GoString() string { return stringName(uint32(i), machineStrings, true) }

// Special section indices.
type SectionIndex int

const (
	SHN_UNDEF     SectionIndex = 0      /* Undefined, missing, irrelevant. */
	SHN_LORESERVE SectionIndex = 0xff00 /* First of reserved range. */
	SHN_LOPROC    SectionIndex = 0xff00 /* First processor-specific. */
	SHN_HIPROC    SectionIndex = 0xff1f /* Last processor-specific. */
	SHN_LOOS      SectionIndex = 0xff20 /* First operating system-specific. */
	SHN_HIOS      SectionIndex = 0xff3f /* Last operating system-specific. */
	SHN_ABS       SectionIndex = 0xfff1 /* Absolute values. */
	SHN_COMMON    SectionIndex = 0xfff2 /* Common data. */
	SHN_XINDEX    SectionIndex = 0xffff /* Escape -- index stored elsewhere. */
	SHN_HIRESERVE SectionIndex = 0xffff /* Last of reserved range. */
)

var shnStrings = []intName{
	{0, "SHN_UNDEF"},
	{0xff00, "SHN_LOPROC"},
	{0xff20, "SHN_LOOS"},
	{0xfff1, "SHN_ABS"},
	{0xfff2, "SHN_COMMON"},
	{0xffff, "SHN_XINDEX"},
}

func (i SectionIndex) String() string   { return stringName(uint32(i), shnStrings, false) }
func (i SectionIndex) GoString() string { return stringName(uint32(i), shnStrings, true) }

// Section type.
type SectionType uint32

const (
	SHT_NULL           SectionType = 0          /* inactive */
	SHT_PROGBITS       SectionType = 1          /* program defined information */
	SHT_SYMTAB         SectionType = 2          /* symbol table section */
	SHT_STRTAB         SectionType = 3          /* string table section */
	SHT_RELA           SectionType = 4          /* relocation section with addends */
	SHT_HASH           SectionType = 5          /* symbol hash table section */
	SHT_DYNAMIC        SectionType = 6          /* dynamic section */
	SHT_NOTE           SectionType = 7          /* note section */
	SHT_NOBITS         SectionType = 8          /* no space section */
	SHT_REL            SectionType = 9          /* relocation section - no addends */
	SHT_SHLIB          SectionType = 10         /* reserved - purpose unknown */
	SHT_DYNSYM         SectionType = 11         /* dynamic symbol table section */
	SHT_INIT_ARRAY     SectionType = 14         /* Initialization function pointers. */
	SHT_FINI_ARRAY     SectionType = 15         /* Termination function pointers. */
	SHT_PREINIT_ARRAY  SectionType = 16         /* Pre-initialization function ptrs. */
	SHT_GROUP          SectionType = 17         /* Section group. */
	SHT_SYMTAB_SHNDX   SectionType = 18         /* Section indexes (see SHN_XINDEX). */
	SHT_LOOS           SectionType = 0x60000000 /* First of OS specific semantics */
	SHT_GNU_ATTRIBUTES SectionType = 0x6ffffff5 /* GNU object attributes */
	SHT_GNU_HASH       SectionType = 0x6ffffff6 /* GNU hash table */
	SHT_GNU_LIBLIST    SectionType = 0x6ffffff7 /* GNU prelink library list */
	SHT_GNU_VERDEF     SectionType = 0x6ffffffd /* GNU version definition section */
	SHT_GNU_VERNEED    SectionType = 0x6ffffffe /* GNU version needs section */
	SHT_GNU_VERSYM     SectionType = 0x6fffffff /* GNU version symbol table */
	SHT_HIOS           SectionType = 0x6fffffff /* Last of OS specific semantics */
	SHT_LOPROC         SectionType = 0x70000000 /* reserved range for processor */
	SHT_HIPROC         SectionType = 0x7fffffff /* specific section header types */
	SHT_LOUSER         SectionType = 0x80000000 /* reserved range for application */
	SHT_HIUSER         SectionType = 0xffffffff /* specific indexes */
)

var shtStrings = []intName{
	{0, "SHT_NULL"},
	{1, "SHT_PROGBITS"},
	{2, "SHT_SYMTAB"},
	{3, "SHT_STRTAB"},
	{4, "SHT_RELA"},
	{5, "SHT_HASH"},
	{6, "SHT_DYNAMIC"},
	{7, "SHT_NOTE"},
	{8, "SHT_NOBITS"},
	{9, "SHT_REL"},
	{10, "SHT_SHLIB"},
	{11, "SHT_DYNSYM"},
	{14, "SHT_INIT_ARRAY"},
	{15, "SHT_FINI_ARRAY"},
	{16, "SHT_PREINIT_ARRAY"},
	{17, "SHT_GROUP"},
	{18, "SHT_SYMTAB_SHNDX"},
	{0x60000000, "SHT_LOOS"},
	{0x6ffffff5, "SHT_GNU_ATTRIBUTES"},
	{0x6ffffff6, "SHT_GNU_HASH"},
	{0x6ffffff7, "SHT_GNU_LIBLIST"},
	{0x6ffffffd, "SHT_GNU_VERDEF"},
	{0x6ffffffe, "SHT_GNU_VERNEED"},
	{0x6fffffff, "SHT_GNU_VERSYM"},
	{0x70000000, "SHT_LOPROC"},
	{0x7fffffff, "SHT_HIPROC"},
	{0x80000000, "SHT_LOUSER"},
	{0xffffffff, "SHT_HIUSER"},
}

func (i SectionType) String() string   { return stringName(uint32(i), shtStrings, false) }
func (i SectionType) GoString() string { return stringName(uint32(i), shtStrings, true) }

// Section flags.
type SectionFlag uint32

const (
	SHF_WRITE            SectionFlag = 0x1        /* Section contains writable data. */
	SHF_ALLOC            SectionFlag = 0x2        /* Section occupies memory. */
	SHF_EXECINSTR        SectionFlag = 0x4        /* Section contains instructions. */
	SHF_MERGE            SectionFlag = 0x10       /* Section may be merged. */
	SHF_STRINGS          SectionFlag = 0x20       /* Section contains strings. */
	SHF_INFO_LINK        SectionFlag = 0x40       /* sh_info holds section index. */
	SHF_LINK_ORDER       SectionFlag = 0x80       /* Special ordering requirements. */
	SHF_OS_NONCONFORMING SectionFlag = 0x100      /* OS-specific processing required. */
	SHF_GROUP            SectionFlag = 0x200      /* Member of section group. */
	SHF_TLS              SectionFlag = 0x400      /* Section contains TLS data. */
	SHF_MASKOS           SectionFlag = 0x0ff00000 /* OS-specific semantics. */
	SHF_MASKPROC         SectionFlag = 0xf0000000 /* Processor-specific semantics. */
)

var shfStrings = []intName{
	{0x1, "SHF_WRITE"},
	{0x2, "SHF_ALLOC"},
	{0x4, "SHF_EXECINSTR"},
	{0x10, "SHF_MERGE"},
	{0x20, "SHF_STRINGS"},
	{0x40, "SHF_INFO_LINK"},
	{0x80, "SHF_LINK_ORDER"},
	{0x100, "SHF_OS_NONCONFORMING"},
	{0x200, "SHF_GROUP"},
	{0x400, "SHF_TLS"},
}

func (i SectionFlag) String() string   { return flagName(uint32(i), shfStrings, false) }
func (i SectionFlag) GoString() string { return flagName(uint32(i), shfStrings, true) }

// Prog.Type
type ProgType int

const (
	PT_NULL    ProgType = 0          /* Unused entry. */
	PT_LOAD    ProgType = 1          /* Loadable segment. */
	PT_DYNAMIC ProgType = 2          /* Dynamic linking information segment. */
	PT_INTERP  ProgType = 3          /* Pathname of interpreter. */
	PT_NOTE    ProgType = 4          /* Auxiliary information. */
	PT_SHLIB   ProgType = 5          /* Reserved (not used). */
	PT_PHDR    ProgType = 6          /* Location of program header itself. */
	PT_TLS     ProgType = 7          /* Thread local storage segment */
	PT_LOOS    ProgType = 0x60000000 /* First OS-specific. */
	PT_HIOS    ProgType = 0x6fffffff /* Last OS-specific. */
	PT_LOPROC  ProgType = 0x70000000 /* First processor-specific type. */
	PT_HIPROC  ProgType = 0x7fffffff /* Last processor-specific type. */
)

var ptStrings = []intName{
	{0, "PT_NULL"},
	{1, "PT_LOAD"},
	{2, "PT_DYNAMIC"},
	{3, "PT_INTERP"},
	{4, "PT_NOTE"},
	{5, "PT_SHLIB"},
	{6, "PT_PHDR"},
	{7, "PT_TLS"},
	{0x60000000, "PT_LOOS"},
	{0x6fffffff, "PT_HIOS"},
	{0x70000000, "PT_LOPROC"},
	{0x7fffffff, "PT_HIPROC"},
}

func (i ProgType) String() string   { return stringName(uint32(i), ptStrings, false) }
func (i ProgType) GoString() string { return stringName(uint32(i), ptStrings, true) }

// Prog.Flag
type ProgFlag uint32

const (
	PF_X        ProgFlag = 0x1        /* Executable. */
	PF_W        ProgFlag = 0x2        /* Writable. */
	PF_R        ProgFlag = 0x4        /* Readable. */
	PF_MASKOS   ProgFlag = 0x0ff00000 /* Operating system-specific. */
	PF_MASKPROC ProgFlag = 0xf0000000 /* Processor-specific. */
)

var pfStrings = []intName{
	{0x1, "PF_X"},
	{0x2, "PF_W"},
	{0x4, "PF_R"},
}

func (i ProgFlag) String() string   { return flagName(uint32(i), pfStrings, false) }
func (i ProgFlag) GoString() string { return flagName(uint32(i), pfStrings, true) }

// Dyn.Tag
type DynTag int

const (
	DT_NULL         DynTag = 0  /* Terminating entry. */
	DT_NEEDED       DynTag = 1  /* String table offset of a needed shared library. */
	DT_PLTRELSZ     DynTag = 2  /* Total size in bytes of PLT relocations. */
	DT_PLTGOT       DynTag = 3  /* Processor-dependent address. */
	DT_HASH         DynTag = 4  /* Address of symbol hash table. */
	DT_STRTAB       DynTag = 5  /* Address of string table. */
	DT_SYMTAB       DynTag = 6  /* Address of symbol table. */
	DT_RELA         DynTag = 7  /* Address of ElfNN_Rela relocations. */
	DT_RELASZ       DynTag = 8  /* Total size of ElfNN_Rela relocations. */
	DT_RELAENT      DynTag = 9  /* Size of each ElfNN_Rela relocation entry. */
	DT_STRSZ        DynTag = 10 /* Size of string table. */
	DT_SYMENT       DynTag = 11 /* Size of each symbol table entry. */
	DT_INIT         DynTag = 12 /* Address of initialization function. */
	DT_FINI         DynTag = 13 /* Address of finalization function. */
	DT_SONAME       DynTag = 14 /* String table offset of shared object name. */
	DT_RPATH        DynTag = 15 /* String table offset of library path. [sup] */
	DT_SYMBOLIC     DynTag = 16 /* Indicates "symbolic" linking. [sup] */
	DT_REL          DynTag = 17 /* Address of ElfNN_Rel relocations. */
	DT_RELSZ        DynTag = 18 /* Total size of ElfNN_Rel relocations. */
	DT_RELENT       DynTag = 19 /* Size of each ElfNN_Rel relocation. */
	DT_PLTREL       DynTag = 20 /* Type of relocation used for PLT. */
	DT_DEBUG        DynTag = 21 /* Reserved (not used). */
	DT_TEXTREL      DynTag = 22 /* Indicates there may be relocations in non-writable segments. [sup] */
	DT_JMPREL       DynTag = 23 /* Address of PLT relocations. */
	DT_BIND_NOW     DynTag = 24 /* [sup] */
	DT_INIT_ARRAY   DynTag = 25 /* Address of the array of pointers to initialization functions */
	DT_FINI_ARRAY   DynTag = 26 /* Address of the array of pointers to termination functions */
	DT_INIT_ARRAYSZ DynTag = 27 /* Size in bytes of the array of initialization functions. */
	DT_FINI_ARRAYSZ DynTag = 28 /* Size in bytes of the array of terminationfunctions. */
	DT_RUNPATH      DynTag = 29 /* String table offset of a null-terminated library search path string. */
	DT_FLAGS        DynTag = 30 /* Object specific flag values. */
	DT_ENCODING     DynTag = 32 /* Values greater than or equal to DT_ENCODING
	   and less than DT_LOOS follow the rules for
	   the interpretation of the d_un union
	   as follows: even == 'd_ptr', even == 'd_val'
	   or none */
	DT_PREINIT_ARRAY   DynTag = 32         /* Address of the array of pointers to pre-initialization functions. */
	DT_PREINIT_ARRAYSZ DynTag = 33         /* Size in bytes of the array of pre-initialization functions. */
	DT_LOOS            DynTag = 0x6000000d /* First OS-specific */
	DT_HIOS            DynTag = 0x6ffff000 /* Last OS-specific */
	DT_VERSYM          DynTag = 0x6ffffff0
	DT_VERNEED         DynTag = 0x6ffffffe
	DT_VERNEEDNUM      DynTag = 0x6fffffff
	DT_LOPROC          DynTag = 0x70000000 /* First processor-specific type. */
	DT_HIPROC          DynTag = 0x7fffffff /* Last processor-specific type. */
)

var dtStrings = []intName{
	{0, "DT_NULL"},
	{1, "DT_NEEDED"},
	{2, "DT_PLTRELSZ"},
	{3, "DT_PLTGOT"},
	{4, "DT_HASH"},
	{5, "DT_STRTAB"},
	{6, "DT_SYMTAB"},
	{7, "DT_RELA"},
	{8, "DT_RELASZ"},
	{9, "DT_RELAENT"},
	{10, "DT_STRSZ"},
	{11, "DT_SYMENT"},
	{12, "DT_INIT"},
	{13, "DT_FINI"},
	{14, "DT_SONAME"},
	{15, "DT_RPATH"},
	{16, "DT_SYMBOLIC"},
	{17, "DT_REL"},
	{18, "DT_RELSZ"},
	{19, "DT_RELENT"},
	{20, "DT_PLTREL"},
	{21, "DT_DEBUG"},
	{22, "DT_TEXTREL"},
	{23, "DT_JMPREL"},
	{24, "DT_BIND_NOW"},
	{25, "DT_INIT_ARRAY"},
	{26, "DT_FINI_ARRAY"},
	{27, "DT_INIT_ARRAYSZ"},
	{28, "DT_FINI_ARRAYSZ"},
	{29, "DT_RUNPATH"},
	{30, "DT_FLAGS"},
	{32, "DT_ENCODING"},
	{32, "DT_PREINIT_ARRAY"},
	{33, "DT_PREINIT_ARRAYSZ"},
	{0x6000000d, "DT_LOOS"},
	{0x6ffff000, "DT_HIOS"},
	{0x6ffffff0, "DT_VERSYM"},
	{0x6ffffffe, "DT_VERNEED"},
	{0x6fffffff, "DT_VERNEEDNUM"},
	{0x70000000, "DT_LOPROC"},
	{0x7fffffff, "DT_HIPROC"},
}

func (i DynTag) String() string   { return stringName(uint32(i), dtStrings, false) }
func (i DynTag) GoString() string { return stringName(uint32(i), dtStrings, true) }

// DT_FLAGS values.
type DynFlag int

const (
	DF_ORIGIN DynFlag = 0x0001 /* Indicates that the object being loaded may
	   make reference to the
	   $ORIGIN substitution string */
	DF_SYMBOLIC DynFlag = 0x0002 /* Indicates "symbolic" linking. */
	DF_TEXTREL  DynFlag = 0x0004 /* Indicates there may be relocations in non-writable segments. */
	DF_BIND_NOW DynFlag = 0x0008 /* Indicates that the dynamic linker should
	   process all relocations for the object
	   containing this entry before transferring
	   control to the program. */
	DF_STATIC_TLS DynFlag = 0x0010 /* Indicates that the shared object or
	   executable contains code using a static
	   thread-local storage scheme. */
)

var dflagStrings = []intName{
	{0x0001, "DF_ORIGIN"},
	{0x0002, "DF_SYMBOLIC"},
	{0x0004, "DF_TEXTREL"},
	{0x0008, "DF_BIND_NOW"},
	{0x0010, "DF_STATIC_TLS"},
}

func (i DynFlag) String() string   { return flagName(uint32(i), dflagStrings, false) }
func (i DynFlag) GoString() string { return flagName(uint32(i), dflagStrings, true) }

// NType values; used in core files.
type NType int

const (
	NT_PRSTATUS NType = 1 /* Process status. */
	NT_FPREGSET NType = 2 /* Floating point registers. */
	NT_PRPSINFO NType = 3 /* Process state info. */
)

var ntypeStrings = []intName{
	{1, "NT_PRSTATUS"},
	{2, "NT_FPREGSET"},
	{3, "NT_PRPSINFO"},
}

func (i NType) String() string   { return stringName(uint32(i), ntypeStrings, false) }
func (i NType) GoString() string { return stringName(uint32(i), ntypeStrings, true) }

/* Symbol Binding - ELFNN_ST_BIND - st_info */
type SymBind int

const (
	STB_LOCAL  SymBind = 0  /* Local symbol */
	STB_GLOBAL SymBind = 1  /* Global symbol */
	STB_WEAK   SymBind = 2  /* like global - lower precedence */
	STB_LOOS   SymBind = 10 /* Reserved range for operating system */
	STB_HIOS   SymBind = 12 /*   specific semantics. */
	STB_LOPROC SymBind = 13 /* reserved range for processor */
	STB_HIPROC SymBind = 15 /*   specific semantics. */
)

var stbStrings = []intName{
	{0, "STB_LOCAL"},
	{1, "STB_GLOBAL"},
	{2, "STB_WEAK"},
	{10, "STB_LOOS"},
	{12, "STB_HIOS"},
	{13, "STB_LOPROC"},
	{15, "STB_HIPROC"},
}

func (i SymBind) String() string   { return stringName(uint32(i), stbStrings, false) }
func (i SymBind) GoString() string { return stringName(uint32(i), stbStrings, true) }

/* Symbol type - ELFNN_ST_TYPE - st_info */
type SymType int

const (
	STT_NOTYPE  SymType = 0  /* Unspecified type. */
	STT_OBJECT  SymType = 1  /* Data object. */
	STT_FUNC    SymType = 2  /* Function. */
	STT_SECTION SymType = 3  /* Section. */
	STT_FILE    SymType = 4  /* Source file. */
	STT_COMMON  SymType = 5  /* Uninitialized common block. */
	STT_TLS     SymType = 6  /* TLS object. */
	STT_LOOS    SymType = 10 /* Reserved range for operating system */
	STT_HIOS    SymType = 12 /*   specific semantics. */
	STT_LOPROC  SymType = 13 /* reserved range for processor */
	STT_HIPROC  SymType = 15 /*   specific semantics. */
)

var sttStrings = []intName{
	{0, "STT_NOTYPE"},
	{1, "STT_OBJECT"},
	{2, "STT_FUNC"},
	{3, "STT_SECTION"},
	{4, "STT_FILE"},
	{5, "STT_COMMON"},
	{6, "STT_TLS"},
	{10, "STT_LOOS"},
	{12, "STT_HIOS"},
	{13, "STT_LOPROC"},
	{15, "STT_HIPROC"},
}

func (i SymType) String() string   { return stringName(uint32(i), sttStrings, false) }
func (i SymType) GoString() string { return stringName(uint32(i), sttStrings, true) }

/* Symbol visibility - ELFNN_ST_VISIBILITY - st_other */
type SymVis int

const (
	STV_DEFAULT   SymVis = 0x0 /* Default visibility (see binding). */
	STV_INTERNAL  SymVis = 0x1 /* Special meaning in relocatable objects. */
	STV_HIDDEN    SymVis = 0x2 /* Not visible. */
	STV_PROTECTED SymVis = 0x3 /* Visible but not preemptible. */
)

var stvStrings = []intName{
	{0x0, "STV_DEFAULT"},
	{0x1, "STV_INTERNAL"},
	{0x2, "STV_HIDDEN"},
	{0x3, "STV_PROTECTED"},
}

func (i SymVis) String() string   { return stringName(uint32(i), stvStrings, false) }
func (i SymVis) GoString() string { return stringName(uint32(i), stvStrings, true) }

/*
 * Relocation types.
 */

// Relocation types for x86-64.
type R_X86_64 int

const (
	R_X86_64_NONE     R_X86_64 = 0  /* No relocation. */
	R_X86_64_64       R_X86_64 = 1  /* Add 64 bit symbol value. */
	R_X86_64_PC32     R_X86_64 = 2  /* PC-relative 32 bit signed sym value. */
	R_X86_64_GOT32    R_X86_64 = 3  /* PC-relative 32 bit GOT offset. */
	R_X86_64_PLT32    R_X86_64 = 4  /* PC-relative 32 bit PLT offset. */
	R_X86_64_COPY     R_X86_64 = 5  /* Copy data from shared object. */
	R_X86_64_GLOB_DAT R_X86_64 = 6  /* Set GOT entry to data address. */
	R_X86_64_JMP_SLOT R_X86_64 = 7  /* Set GOT entry to code address. */
	R_X86_64_RELATIVE R_X86_64 = 8  /* Add load address of shared object. */
	R_X86_64_GOTPCREL R_X86_64 = 9  /* Add 32 bit signed pcrel offset to GOT. */
	R_X86_64_32       R_X86_64 = 10 /* Add 32 bit zero extended symbol value */
	R_X86_64_32S      R_X86_64 = 11 /* Add 32 bit sign extended symbol value */
	R_X86_64_16       R_X86_64 = 12 /* Add 16 bit zero extended symbol value */
	R_X86_64_PC16     R_X86_64 = 13 /* Add 16 bit signed extended pc relative symbol value */
	R_X86_64_8        R_X86_64 = 14 /* Add 8 bit zero extended symbol value */
	R_X86_64_PC8      R_X86_64 = 15 /* Add 8 bit signed extended pc relative symbol value */
	R_X86_64_DTPMOD64 R_X86_64 = 16 /* ID of module containing symbol */
	R_X86_64_DTPOFF64 R_X86_64 = 17 /* Offset in TLS block */
	R_X86_64_TPOFF64  R_X86_64 = 18 /* Offset in static TLS block */
	R_X86_64_TLSGD    R_X86_64 = 19 /* PC relative offset to GD GOT entry */
	R_X86_64_TLSLD    R_X86_64 = 20 /* PC relative offset to LD GOT entry */
	R_X86_64_DTPOFF32 R_X86_64 = 21 /* Offset in TLS block */
	R_X86_64_GOTTPOFF R_X86_64 = 22 /* PC relative offset to IE GOT entry */
	R_X86_64_TPOFF32  R_X86_64 = 23 /* Offset in static TLS block */
)

var rx86_64Strings = []intName{
	{0, "R_X86_64_NONE"},
	{1, "R_X86_64_64"},
	{2, "R_X86_64_PC32"},
	{3, "R_X86_64_GOT32"},
	{4, "R_X86_64_PLT32"},
	{5, "R_X86_64_COPY"},
	{6, "R_X86_64_GLOB_DAT"},
	{7, "R_X86_64_JMP_SLOT"},
	{8, "R_X86_64_RELATIVE"},
	{9, "R_X86_64_GOTPCREL"},
	{10, "R_X86_64_32"},
	{11, "R_X86_64_32S"},
	{12, "R_X86_64_16"},
	{13, "R_X86_64_PC16"},
	{14, "R_X86_64_8"},
	{15, "R_X86_64_PC8"},
	{16, "R_X86_64_DTPMOD64"},
	{17, "R_X86_64_DTPOFF64"},
	{18, "R_X86_64_TPOFF64"},
	{19, "R_X86_64_TLSGD"},
	{20, "R_X86_64_TLSLD"},
	{21, "R_X86_64_DTPOFF32"},
	{22, "R_X86_64_GOTTPOFF"},
	{23, "R_X86_64_TPOFF32"},
}

func (i R_X86_64) String() string   { return stringName(uint32(i), rx86_64Strings, false) }
func (i R_X86_64) GoString() string { return stringName(uint32(i), rx86_64Strings, true) }

// Relocation types for Alpha.
type R_ALPHA int

const (
	R_ALPHA_NONE           R_ALPHA = 0  /* No reloc */
	R_ALPHA_REFLONG        R_ALPHA = 1  /* Direct 32 bit */
	R_ALPHA_REFQUAD        R_ALPHA = 2  /* Direct 64 bit */
	R_ALPHA_GPREL32        R_ALPHA = 3  /* GP relative 32 bit */
	R_ALPHA_LITERAL        R_ALPHA = 4  /* GP relative 16 bit w/optimization */
	R_ALPHA_LITUSE         R_ALPHA = 5  /* Optimization hint for LITERAL */
	R_ALPHA_GPDISP         R_ALPHA = 6  /* Add displacement to GP */
	R_ALPHA_BRADDR         R_ALPHA = 7  /* PC+4 relative 23 bit shifted */
	R_ALPHA_HINT           R_ALPHA = 8  /* PC+4 relative 16 bit shifted */
	R_ALPHA_SREL16         R_ALPHA = 9  /* PC relative 16 bit */
	R_ALPHA_SREL32         R_ALPHA = 10 /* PC relative 32 bit */
	R_ALPHA_SREL64         R_ALPHA = 11 /* PC relative 64 bit */
	R_ALPHA_OP_PUSH        R_ALPHA = 12 /* OP stack push */
	R_ALPHA_OP_STORE       R_ALPHA = 13 /* OP stack pop and store */
	R_ALPHA_OP_PSUB        R_ALPHA = 14 /* OP stack subtract */
	R_ALPHA_OP_PRSHIFT     R_ALPHA = 15 /* OP stack right shift */
	R_ALPHA_GPVALUE        R_ALPHA = 16
	R_ALPHA_GPRELHIGH      R_ALPHA = 17
	R_ALPHA_GPRELLOW       R_ALPHA = 18
	R_ALPHA_IMMED_GP_16    R_ALPHA = 19
	R_ALPHA_IMMED_GP_HI32  R_ALPHA = 20
	R_ALPHA_IMMED_SCN_HI32 R_ALPHA = 21
	R_ALPHA_IMMED_BR_HI32  R_ALPHA = 22
	R_ALPHA_IMMED_LO32     R_ALPHA = 23
	R_ALPHA_COPY           R_ALPHA = 24 /* Copy symbol at runtime */
	R_ALPHA_GLOB_DAT       R_ALPHA = 25 /* Create GOT entry */
	R_ALPHA_JMP_SLOT       R_ALPHA = 26 /* Create PLT entry */
	R_ALPHA_RELATIVE       R_ALPHA = 27 /* Adjust by program base */
)

var ralphaStrings = []intName{
	{0, "R_ALPHA_NONE"},
	{1, "R_ALPHA_REFLONG"},
	{2, "R_ALPHA_REFQUAD"},
	{3, "R_ALPHA_GPREL32"},
	{4, "R_ALPHA_LITERAL"},
	{5, "R_ALPHA_LITUSE"},
	{6, "R_ALPHA_GPDISP"},
	{7, "R_ALPHA_BRADDR"},
	{8, "R_ALPHA_HINT"},
	{9, "R_ALPHA_SREL16"},
	{10, "R_ALPHA_SREL32"},
	{11, "R_ALPHA_SREL64"},
	{12, "R_ALPHA_OP_PUSH"},
	{13, "R_ALPHA_OP_STORE"},
	{14, "R_ALPHA_OP_PSUB"},
	{15, "R_ALPHA_OP_PRSHIFT"},
	{16, "R_ALPHA_GPVALUE"},
	{17, "R_ALPHA_GPRELHIGH"},
	{18, "R_ALPHA_GPRELLOW"},
	{19, "R_ALPHA_IMMED_GP_16"},
	{20, "R_ALPHA_IMMED_GP_HI32"},
	{21, "R_ALPHA_IMMED_SCN_HI32"},
	{22, "R_ALPHA_IMMED_BR_HI32"},
	{23, "R_ALPHA_IMMED_LO32"},
	{24, "R_ALPHA_COPY"},
	{25, "R_ALPHA_GLOB_DAT"},
	{26, "R_ALPHA_JMP_SLOT"},
	{27, "R_ALPHA_RELATIVE"},
}

func (i R_ALPHA) String() string   { return stringName(uint32(i), ralphaStrings, false) }
func (i R_ALPHA) GoString() string { return stringName(uint32(i), ralphaStrings, true) }

// Relocation types for ARM.
type R_ARM int

const (
	R_ARM_NONE          R_ARM = 0 /* No relocation. */
	R_ARM_PC24          R_ARM = 1
	R_ARM_ABS32         R_ARM = 2
	R_ARM_REL32         R_ARM = 3
	R_ARM_PC13          R_ARM = 4
	R_ARM_ABS16         R_ARM = 5
	R_ARM_ABS12         R_ARM = 6
	R_ARM_THM_ABS5      R_ARM = 7
	R_ARM_ABS8          R_ARM = 8
	R_ARM_SBREL32       R_ARM = 9
	R_ARM_THM_PC22      R_ARM = 10
	R_ARM_THM_PC8       R_ARM = 11
	R_ARM_AMP_VCALL9    R_ARM = 12
	R_ARM_SWI24         R_ARM = 13
	R_ARM_THM_SWI8      R_ARM = 14
	R_ARM_XPC25         R_ARM = 15
	R_ARM_THM_XPC22     R_ARM = 16
	R_ARM_COPY          R_ARM = 20 /* Copy data from shared object. */
	R_ARM_GLOB_DAT      R_ARM = 21 /* Set GOT entry to data address. */
	R_ARM_JUMP_SLOT     R_ARM = 22 /* Set GOT entry to code address. */
	R_ARM_RELATIVE      R_ARM = 23 /* Add load address of shared object. */
	R_ARM_GOTOFF        R_ARM = 24 /* Add GOT-relative symbol address. */
	R_ARM_GOTPC         R_ARM = 25 /* Add PC-relative GOT table address. */
	R_ARM_GOT32         R_ARM = 26 /* Add PC-relative GOT offset. */
	R_ARM_PLT32         R_ARM = 27 /* Add PC-relative PLT offset. */
	R_ARM_GNU_VTENTRY   R_ARM = 100
	R_ARM_GNU_VTINHERIT R_ARM = 101
	R_ARM_RSBREL32      R_ARM = 250
	R_ARM_THM_RPC22     R_ARM = 251
	R_ARM_RREL32        R_ARM = 252
	R_ARM_RABS32        R_ARM = 253
	R_ARM_RPC24         R_ARM = 254
	R_ARM_RBASE         R_ARM = 255
)

var rarmStrings = []intName{
	{0, "R_ARM_NONE"},
	{1, "R_ARM_PC24"},
	{2, "R_ARM_ABS32"},
	{3, "R_ARM_REL32"},
	{4, "R_ARM_PC13"},
	{5, "R_ARM_ABS16"},
	{6, "R_ARM_ABS12"},
	{7, "R_ARM_THM_ABS5"},
	{8, "R_ARM_ABS8"},
	{9, "R_ARM_SBREL32"},
	{10, "R_ARM_THM_PC22"},
	{11, "R_ARM_THM_PC8"},
	{12, "R_ARM_AMP_VCALL9"},
	{13, "R_ARM_SWI24"},
	{14, "R_ARM_THM_SWI8"},
	{15, "R_ARM_XPC25"},
	{16, "R_ARM_THM_XPC22"},
	{20, "R_ARM_COPY"},
	{21, "R_ARM_GLOB_DAT"},
	{22, "R_ARM_JUMP_SLOT"},
	{23, "R_ARM_RELATIVE"},
	{24, "R_ARM_GOTOFF"},
	{25, "R_ARM_GOTPC"},
	{26, "R_ARM_GOT32"},
	{27, "R_ARM_PLT32"},
	{100, "R_ARM_GNU_VTENTRY"},
	{101, "R_ARM_GNU_VTINHERIT"},
	{250, "R_ARM_RSBREL32"},
	{251, "R_ARM_THM_RPC22"},
	{252, "R_ARM_RREL32"},
	{253, "R_ARM_RABS32"},
	{254, "R_ARM_RPC24"},
	{255, "R_ARM_RBASE"},
}

func (i R_ARM) String() string   { return stringName(uint32(i), rarmStrings, false) }
func (i R_ARM) GoString() string { return stringName(uint32(i), rarmStrings, true) }

// Relocation types for 386.
type R_386 int

const (
	R_386_NONE         R_386 = 0  /* No relocation. */
	R_386_32           R_386 = 1  /* Add symbol value. */
	R_386_PC32         R_386 = 2  /* Add PC-relative symbol value. */
	R_386_GOT32        R_386 = 3  /* Add PC-relative GOT offset. */
	R_386_PLT32        R_386 = 4  /* Add PC-relative PLT offset. */
	R_386_COPY         R_386 = 5  /* Copy data from shared object. */
	R_386_GLOB_DAT     R_386 = 6  /* Set GOT entry to data address. */
	R_386_JMP_SLOT     R_386 = 7  /* Set GOT entry to code address. */
	R_386_RELATIVE     R_386 = 8  /* Add load address of shared object. */
	R_386_GOTOFF       R_386 = 9  /* Add GOT-relative symbol address. */
	R_386_GOTPC        R_386 = 10 /* Add PC-relative GOT table address. */
	R_386_TLS_TPOFF    R_386 = 14 /* Negative offset in static TLS block */
	R_386_TLS_IE       R_386 = 15 /* Absolute address of GOT for -ve static TLS */
	R_386_TLS_GOTIE    R_386 = 16 /* GOT entry for negative static TLS block */
	R_386_TLS_LE       R_386 = 17 /* Negative offset relative to static TLS */
	R_386_TLS_GD       R_386 = 18 /* 32 bit offset to GOT (index,off) pair */
	R_386_TLS_LDM      R_386 = 19 /* 32 bit offset to GOT (index,zero) pair */
	R_386_TLS_GD_32    R_386 = 24 /* 32 bit offset to GOT (index,off) pair */
	R_386_TLS_GD_PUSH  R_386 = 25 /* pushl instruction for Sun ABI GD sequence */
	R_386_TLS_GD_CALL  R_386 = 26 /* call instruction for Sun ABI GD sequence */
	R_386_TLS_GD_POP   R_386 = 27 /* popl instruction for Sun ABI GD sequence */
	R_386_TLS_LDM_32   R_386 = 28 /* 32 bit offset to GOT (index,zero) pair */
	R_386_TLS_LDM_PUSH R_386 = 29 /* pushl instruction for Sun ABI LD sequence */
	R_386_TLS_LDM_CALL R_386 = 30 /* call instruction for Sun ABI LD sequence */
	R_386_TLS_LDM_POP  R_386 = 31 /* popl instruction for Sun ABI LD sequence */
	R_386_TLS_LDO_32   R_386 = 32 /* 32 bit offset from start of TLS block */
	R_386_TLS_IE_32    R_386 = 33 /* 32 bit offset to GOT static TLS offset entry */
	R_386_TLS_LE_32    R_386 = 34 /* 32 bit offset within static TLS block */
	R_386_TLS_DTPMOD32 R_386 = 35 /* GOT entry containing TLS index */
	R_386_TLS_DTPOFF32 R_386 = 36 /* GOT entry containing TLS offset */
	R_386_TLS_TPOFF32  R_386 = 37 /* GOT entry of -ve static TLS offset */
)

var r386Strings = []intName{
	{0, "R_386_NONE"},
	{1, "R_386_32"},
	{2, "R_386_PC32"},
	{3, "R_386_GOT32"},
	{4, "R_386_PLT32"},
	{5, "R_386_COPY"},
	{6, "R_386_GLOB_DAT"},
	{7, "R_386_JMP_SLOT"},
	{8, "R_386_RELATIVE"},
	{9, "R_386_GOTOFF"},
	{10, "R_386_GOTPC"},
	{14, "R_386_TLS_TPOFF"},
	{15, "R_386_TLS_IE"},
	{16, "R_386_TLS_GOTIE"},
	{17, "R_386_TLS_LE"},
	{18, "R_386_TLS_GD"},
	{19, "R_386_TLS_LDM"},
	{24, "R_386_TLS_GD_32"},
	{25, "R_386_TLS_GD_PUSH"},
	{26, "R_386_TLS_GD_CALL"},
	{27, "R_386_TLS_GD_POP"},
	{28, "R_386_TLS_LDM_32"},
	{29, "R_386_TLS_LDM_PUSH"},
	{30, "R_386_TLS_LDM_CALL"},
	{31, "R_386_TLS_LDM_POP"},
	{32, "R_386_TLS_LDO_32"},
	{33, "R_386_TLS_IE_32"},
	{34, "R_386_TLS_LE_32"},
	{35, "R_386_TLS_DTPMOD32"},
	{36, "R_386_TLS_DTPOFF32"},
	{37, "R_386_TLS_TPOFF32"},
}

func (i R_386) String() string   { return stringName(uint32(i), r386Strings, false) }
func (i R_386) GoString() string { return stringName(uint32(i), r386Strings, true) }

// Relocation types for PowerPC.
type R_PPC int

const (
	R_PPC_NONE            R_PPC = 0 /* No relocation. */
	R_PPC_ADDR32          R_PPC = 1
	R_PPC_ADDR24          R_PPC = 2
	R_PPC_ADDR16          R_PPC = 3
	R_PPC_ADDR16_LO       R_PPC = 4
	R_PPC_ADDR16_HI       R_PPC = 5
	R_PPC_ADDR16_HA       R_PPC = 6
	R_PPC_ADDR14          R_PPC = 7
	R_PPC_ADDR14_BRTAKEN  R_PPC = 8
	R_PPC_ADDR14_BRNTAKEN R_PPC = 9
	R_PPC_REL24           R_PPC = 10
	R_PPC_REL14           R_PPC = 11
	R_PPC_REL14_BRTAKEN   R_PPC = 12
	R_PPC_REL14_BRNTAKEN  R_PPC = 13
	R_PPC_GOT16           R_PPC = 14
	R_PPC_GOT16_LO        R_PPC = 15
	R_PPC_GOT16_HI        R_PPC = 16
	R_PPC_GOT16_HA        R_PPC = 17
	R_PPC_PLTREL24        R_PPC = 18
	R_PPC_COPY            R_PPC = 19
	R_PPC_GLOB_DAT        R_PPC = 20
	R_PPC_JMP_SLOT        R_PPC = 21
	R_PPC_RELATIVE        R_PPC = 22
	R_PPC_LOCAL24PC       R_PPC = 23
	R_PPC_UADDR32         R_PPC = 24
	R_PPC_UADDR16         R_PPC = 25
	R_PPC_REL32           R_PPC = 26
	R_PPC_PLT32           R_PPC = 27
	R_PPC_PLTREL32        R_PPC = 28
	R_PPC_PLT16_LO        R_PPC = 29
	R_PPC_PLT16_HI        R_PPC = 30
	R_PPC_PLT16_HA        R_PPC = 31
	R_PPC_SDAREL16        R_PPC = 32
	R_PPC_SECTOFF         R_PPC = 33
	R_PPC_SECTOFF_LO      R_PPC = 34
	R_PPC_SECTOFF_HI      R_PPC = 35
	R_PPC_SECTOFF_HA      R_PPC = 36
	R_PPC_TLS             R_PPC = 67
	R_PPC_DTPMOD32        R_PPC = 68
	R_PPC_TPREL16         R_PPC = 69
	R_PPC_TPREL16_LO      R_PPC = 70
	R_PPC_TPREL16_HI      R_PPC = 71
	R_PPC_TPREL16_HA      R_PPC = 72
	R_PPC_TPREL32         R_PPC = 73
	R_PPC_DTPREL16        R_PPC = 74
	R_PPC_DTPREL16_LO     R_PPC = 75
	R_PPC_DTPREL16_HI     R_PPC = 76
	R_PPC_DTPREL16_HA     R_PPC = 77
	R_PPC_DTPREL32        R_PPC = 78
	R_PPC_GOT_TLSGD16     R_PPC = 79
	R_PPC_GOT_TLSGD16_LO  R_PPC = 80
	R_PPC_GOT_TLSGD16_HI  R_PPC = 81
	R_PPC_GOT_TLSGD16_HA  R_PPC = 82
	R_PPC_GOT_TLSLD16     R_PPC = 83
	R_PPC_GOT_TLSLD16_LO  R_PPC = 84
	R_PPC_GOT_TLSLD16_HI  R_PPC = 85
	R_PPC_GOT_TLSLD16_HA  R_PPC = 86
	R_PPC_GOT_TPREL16     R_PPC = 87
	R_PPC_GOT_TPREL16_LO  R_PPC = 88
	R_PPC_GOT_TPREL16_HI  R_PPC = 89
	R_PPC_GOT_TPREL16_HA  R_PPC = 90
	R_PPC_EMB_NADDR32     R_PPC = 101
	R_PPC_EMB_NADDR16     R_PPC = 102
	R_PPC_EMB_NADDR16_LO  R_PPC = 103
	R_PPC_EMB_NADDR16_HI  R_PPC = 104
	R_PPC_EMB_NADDR16_HA  R_PPC = 105
	R_PPC_EMB_SDAI16      R_PPC = 106
	R_PPC_EMB_SDA2I16     R_PPC = 107
	R_PPC_EMB_SDA2REL     R_PPC = 108
	R_PPC_EMB_SDA21       R_PPC = 109
	R_PPC_EMB_MRKREF      R_PPC = 110
	R_PPC_EMB_RELSEC16    R_PPC = 111
	R_PPC_EMB_RELST_LO    R_PPC = 112
	R_PPC_EMB_RELST_HI    R_PPC = 113
	R_PPC_EMB_RELST_HA    R_PPC = 114
	R_PPC_EMB_BIT_FLD     R_PPC = 115
	R_PPC_EMB_RELSDA      R_PPC = 116
)

var rppcStrings = []intName{
	{0, "R_PPC_NONE"},
	{1, "R_PPC_ADDR32"},
	{2, "R_PPC_ADDR24"},
	{3, "R_PPC_ADDR16"},
	{4, "R_PPC_ADDR16_LO"},
	{5, "R_PPC_ADDR16_HI"},
	{6, "R_PPC_ADDR16_HA"},
	{7, "R_PPC_ADDR14"},
	{8, "R_PPC_ADDR14_BRTAKEN"},
	{9, "R_PPC_ADDR14_BRNTAKEN"},
	{10, "R_PPC_REL24"},
	{11, "R_PPC_REL14"},
	{12, "R_PPC_REL14_BRTAKEN"},
	{13, "R_PPC_REL14_BRNTAKEN"},
	{14, "R_PPC_GOT16"},
	{15, "R_PPC_GOT16_LO"},
	{16, "R_PPC_GOT16_HI"},
	{17, "R_PPC_GOT16_HA"},
	{18, "R_PPC_PLTREL24"},
	{19, "R_PPC_COPY"},
	{20, "R_PPC_GLOB_DAT"},
	{21, "R_PPC_JMP_SLOT"},
	{22, "R_PPC_RELATIVE"},
	{23, "R_PPC_LOCAL24PC"},
	{24, "R_PPC_UADDR32"},
	{25, "R_PPC_UADDR16"},
	{26, "R_PPC_REL32"},
	{27, "R_PPC_PLT32"},
	{28, "R_PPC_PLTREL32"},
	{29, "R_PPC_PLT16_LO"},
	{30, "R_PPC_PLT16_HI"},
	{31, "R_PPC_PLT16_HA"},
	{32, "R_PPC_SDAREL16"},
	{33, "R_PPC_SECTOFF"},
	{34, "R_PPC_SECTOFF_LO"},
	{35, "R_PPC_SECTOFF_HI"},
	{36, "R_PPC_SECTOFF_HA"},

	{67, "R_PPC_TLS"},
	{68, "R_PPC_DTPMOD32"},
	{69, "R_PPC_TPREL16"},
	{70, "R_PPC_TPREL16_LO"},
	{71, "R_PPC_TPREL16_HI"},
	{72, "R_PPC_TPREL16_HA"},
	{73, "R_PPC_TPREL32"},
	{74, "R_PPC_DTPREL16"},
	{75, "R_PPC_DTPREL16_LO"},
	{76, "R_PPC_DTPREL16_HI"},
	{77, "R_PPC_DTPREL16_HA"},
	{78, "R_PPC_DTPREL32"},
	{79, "R_PPC_GOT_TLSGD16"},
	{80, "R_PPC_GOT_TLSGD16_LO"},
	{81, "R_PPC_GOT_TLSGD16_HI"},
	{82, "R_PPC_GOT_TLSGD16_HA"},
	{83, "R_PPC_GOT_TLSLD16"},
	{84, "R_PPC_GOT_TLSLD16_LO"},
	{85, "R_PPC_GOT_TLSLD16_HI"},
	{86, "R_PPC_GOT_TLSLD16_HA"},
	{87, "R_PPC_GOT_TPREL16"},
	{88, "R_PPC_GOT_TPREL16_LO"},
	{89, "R_PPC_GOT_TPREL16_HI"},
	{90, "R_PPC_GOT_TPREL16_HA"},

	{101, "R_PPC_EMB_NADDR32"},
	{102, "R_PPC_EMB_NADDR16"},
	{103, "R_PPC_EMB_NADDR16_LO"},
	{104, "R_PPC_EMB_NADDR16_HI"},
	{105, "R_PPC_EMB_NADDR16_HA"},
	{106, "R_PPC_EMB_SDAI16"},
	{107, "R_PPC_EMB_SDA2I16"},
	{108, "R_PPC_EMB_SDA2REL"},
	{109, "R_PPC_EMB_SDA21"},
	{110, "R_PPC_EMB_MRKREF"},
	{111, "R_PPC_EMB_RELSEC16"},
	{112, "R_PPC_EMB_RELST_LO"},
	{113, "R_PPC_EMB_RELST_HI"},
	{114, "R_PPC_EMB_RELST_HA"},
	{115, "R_PPC_EMB_BIT_FLD"},
	{116, "R_PPC_EMB_RELSDA"},
}

func (i R_PPC) String() string   { return stringName(uint32(i), rppcStrings, false) }
func (i R_PPC) GoString() string { return stringName(uint32(i), rppcStrings, true) }

// Relocation types for SPARC.
type R_SPARC int

const (
	R_SPARC_NONE     R_SPARC = 0
	R_SPARC_8        R_SPARC = 1
	R_SPARC_16       R_SPARC = 2
	R_SPARC_32       R_SPARC = 3
	R_SPARC_DISP8    R_SPARC = 4
	R_SPARC_DISP16   R_SPARC = 5
	R_SPARC_DISP32   R_SPARC = 6
	R_SPARC_WDISP30  R_SPARC = 7
	R_SPARC_WDISP22  R_SPARC = 8
	R_SPARC_HI22     R_SPARC = 9
	R_SPARC_22       R_SPARC = 10
	R_SPARC_13       R_SPARC = 11
	R_SPARC_LO10     R_SPARC = 12
	R_SPARC_GOT10    R_SPARC = 13
	R_SPARC_GOT13    R_SPARC = 14
	R_SPARC_GOT22    R_SPARC = 15
	R_SPARC_PC10     R_SPARC = 16
	R_SPARC_PC22     R_SPARC = 17
	R_SPARC_WPLT30   R_SPARC = 18
	R_SPARC_COPY     R_SPARC = 19
	R_SPARC_GLOB_DAT R_SPARC = 20
	R_SPARC_JMP_SLOT R_SPARC = 21
	R_SPARC_RELATIVE R_SPARC = 22
	R_SPARC_UA32     R_SPARC = 23
	R_SPARC_PLT32    R_SPARC = 24
	R_SPARC_HIPLT22  R_SPARC = 25
	R_SPARC_LOPLT10  R_SPARC = 26
	R_SPARC_PCPLT32  R_SPARC = 27
	R_SPARC_PCPLT22  R_SPARC = 28
	R_SPARC_PCPLT10  R_SPARC = 29
	R_SPARC_10       R_SPARC = 30
	R_SPARC_11       R_SPARC = 31
	R_SPARC_64       R_SPARC = 32
	R_SPARC_OLO10    R_SPARC = 33
	R_SPARC_HH22     R_SPARC = 34
	R_SPARC_HM10     R_SPARC = 35
	R_SPARC_LM22     R_SPARC = 36
	R_SPARC_PC_HH22  R_SPARC = 37
	R_SPARC_PC_HM10  R_SPARC = 38
	R_SPARC_PC_LM22  R_SPARC = 39
	R_SPARC_WDISP16  R_SPARC = 40
	R_SPARC_WDISP19  R_SPARC = 41
	R_SPARC_GLOB_JMP R_SPARC = 42
	R_SPARC_7        R_SPARC = 43
	R_SPARC_5        R_SPARC = 44
	R_SPARC_6        R_SPARC = 45
	R_SPARC_DISP64   R_SPARC = 46
	R_SPARC_PLT64    R_SPARC = 47
	R_SPARC_HIX22    R_SPARC = 48
	R_SPARC_LOX10    R_SPARC = 49
	R_SPARC_H44      R_SPARC = 50
	R_SPARC_M44      R_SPARC = 51
	R_SPARC_L44      R_SPARC = 52
	R_SPARC_REGISTER R_SPARC = 53
	R_SPARC_UA64     R_SPARC = 54
	R_SPARC_UA16     R_SPARC = 55
)

var rsparcStrings = []intName{
	{0, "R_SPARC_NONE"},
	{1, "R_SPARC_8"},
	{2, "R_SPARC_16"},
	{3, "R_SPARC_32"},
	{4, "R_SPARC_DISP8"},
	{5, "R_SPARC_DISP16"},
	{6, "R_SPARC_DISP32"},
	{7, "R_SPARC_WDISP30"},
	{8, "R_SPARC_WDISP22"},
	{9, "R_SPARC_HI22"},
	{10, "R_SPARC_22"},
	{11, "R_SPARC_13"},
	{12, "R_SPARC_LO10"},
	{13, "R_SPARC_GOT10"},
	{14, "R_SPARC_GOT13"},
	{15, "R_SPARC_GOT22"},
	{16, "R_SPARC_PC10"},
	{17, "R_SPARC_PC22"},
	{18, "R_SPARC_WPLT30"},
	{19, "R_SPARC_COPY"},
	{20, "R_SPARC_GLOB_DAT"},
	{21, "R_SPARC_JMP_SLOT"},
	{22, "R_SPARC_RELATIVE"},
	{23, "R_SPARC_UA32"},
	{24, "R_SPARC_PLT32"},
	{25, "R_SPARC_HIPLT22"},
	{26, "R_SPARC_LOPLT10"},
	{27, "R_SPARC_PCPLT32"},
	{28, "R_SPARC_PCPLT22"},
	{29, "R_SPARC_PCPLT10"},
	{30, "R_SPARC_10"},
	{31, "R_SPARC_11"},
	{32, "R_SPARC_64"},
	{33, "R_SPARC_OLO10"},
	{34, "R_SPARC_HH22"},
	{35, "R_SPARC_HM10"},
	{36, "R_SPARC_LM22"},
	{37, "R_SPARC_PC_HH22"},
	{38, "R_SPARC_PC_HM10"},
	{39, "R_SPARC_PC_LM22"},
	{40, "R_SPARC_WDISP16"},
	{41, "R_SPARC_WDISP19"},
	{42, "R_SPARC_GLOB_JMP"},
	{43, "R_SPARC_7"},
	{44, "R_SPARC_5"},
	{45, "R_SPARC_6"},
	{46, "R_SPARC_DISP64"},
	{47, "R_SPARC_PLT64"},
	{48, "R_SPARC_HIX22"},
	{49, "R_SPARC_LOX10"},
	{50, "R_SPARC_H44"},
	{51, "R_SPARC_M44"},
	{52, "R_SPARC_L44"},
	{53, "R_SPARC_REGISTER"},
	{54, "R_SPARC_UA64"},
	{55, "R_SPARC_UA16"},
}

func (i R_SPARC) String() string   { return stringName(uint32(i), rsparcStrings, false) }
func (i R_SPARC) GoString() string { return stringName(uint32(i), rsparcStrings, true) }

// Magic number for the elf trampoline, chosen wisely to be an immediate value.
const ARM_MAGIC_TRAMP_NUMBER = 0x5c000003

// ELF32 File header.
type Header32 struct {
	Ident     [EI_NIDENT]byte /* File identification. */
	Type      uint16          /* File type. */
	Machine   uint16          /* Machine architecture. */
	Version   uint32          /* ELF format version. */
	Entry     uint32          /* Entry point. */
	Phoff     uint32          /* Program header file offset. */
	Shoff     uint32          /* Section header file offset. */
	Flags     uint32          /* Architecture-specific flags. */
	Ehsize    uint16          /* Size of ELF header in bytes. */
	Phentsize uint16          /* Size of program header entry. */
	Phnum     uint16          /* Number of program header entries. */
	Shentsize uint16          /* Size of section header entry. */
	Shnum     uint16          /* Number of section header entries. */
	Shstrndx  uint16          /* Section name strings section. */
}

// ELF32 Section header.
type Section32 struct {
	Name      uint32 /* Section name (index into the section header string table). */
	Type      uint32 /* Section type. */
	Flags     uint32 /* Section flags. */
	Addr      uint32 /* Address in memory image. */
	Off       uint32 /* Offset in file. */
	Size      uint32 /* Size in bytes. */
	Link      uint32 /* Index of a related section. */
	Info      uint32 /* Depends on section type. */
	Addralign uint32 /* Alignment in bytes. */
	Entsize   uint32 /* Size of each entry in section. */
}

// ELF32 Program header.
type Prog32 struct {
	Type   uint32 /* Entry type. */
	Off    uint32 /* File offset of contents. */
	Vaddr  uint32 /* Virtual address in memory image. */
	Paddr  uint32 /* Physical address (not used). */
	Filesz uint32 /* Size of contents in file. */
	Memsz  uint32 /* Size of contents in memory. */
	Flags  uint32 /* Access permission flags. */
	Align  uint32 /* Alignment in memory and file. */
}

// ELF32 Dynamic structure.  The ".dynamic" section contains an array of them.
type Dyn32 struct {
	Tag int32  /* Entry type. */
	Val uint32 /* Integer/Address value. */
}

/*
 * Relocation entries.
 */

// ELF32 Relocations that don't need an addend field.
type Rel32 struct {
	Off  uint32 /* Location to be relocated. */
	Info uint32 /* Relocation type and symbol index. */
}

// ELF32 Relocations that need an addend field.
type Rela32 struct {
	Off    uint32 /* Location to be relocated. */
	Info   uint32 /* Relocation type and symbol index. */
	Addend int32  /* Addend. */
}

func R_SYM32(info uint32) uint32      { return uint32(info >> 8) }
func R_TYPE32(info uint32) uint32     { return uint32(info & 0xff) }
func R_INFO32(sym, typ uint32) uint32 { return sym<<8 | typ }

// ELF32 Symbol.
type Sym32 struct {
	Name  uint32
	Value uint32
	Size  uint32
	Info  uint8
	Other uint8
	Shndx uint16
}

const Sym32Size = 16

func ST_BIND(info uint8) SymBind { return SymBind(info >> 4) }
func ST_TYPE(info uint8) SymType { return SymType(info & 0xF) }
func ST_INFO(bind SymBind, typ SymType) uint8 {
	return uint8(bind)<<4 | uint8(typ)&0xf
}
func ST_VISIBILITY(other uint8) SymVis { return SymVis(other & 3) }

/*
 * ELF64
 */

// ELF64 file header.
type Header64 struct {
	Ident     [EI_NIDENT]byte /* File identification. */
	Type      uint16          /* File type. */
	Machine   uint16          /* Machine architecture. */
	Version   uint32          /* ELF format version. */
	Entry     uint64          /* Entry point. */
	Phoff     uint64          /* Program header file offset. */
	Shoff     uint64          /* Section header file offset. */
	Flags     uint32          /* Architecture-specific flags. */
	Ehsize    uint16          /* Size of ELF header in bytes. */
	Phentsize uint16          /* Size of program header entry. */
	Phnum     uint16          /* Number of program header entries. */
	Shentsize uint16          /* Size of section header entry. */
	Shnum     uint16          /* Number of section header entries. */
	Shstrndx  uint16          /* Section name strings section. */
}

// ELF64 Section header.
type Section64 struct {
	Name      uint32 /* Section name (index into the section header string table). */
	Type      uint32 /* Section type. */
	Flags     uint64 /* Section flags. */
	Addr      uint64 /* Address in memory image. */
	Off       uint64 /* Offset in file. */
	Size      uint64 /* Size in bytes. */
	Link      uint32 /* Index of a related section. */
	Info      uint32 /* Depends on section type. */
	Addralign uint64 /* Alignment in bytes. */
	Entsize   uint64 /* Size of each entry in section. */
}

// ELF64 Program header.
type Prog64 struct {
	Type   uint32 /* Entry type. */
	Flags  uint32 /* Access permission flags. */
	Off    uint64 /* File offset of contents. */
	Vaddr  uint64 /* Virtual address in memory image. */
	Paddr  uint64 /* Physical address (not used). */
	Filesz uint64 /* Size of contents in file. */
	Memsz  uint64 /* Size of contents in memory. */
	Align  uint64 /* Alignment in memory and file. */
}

// ELF64 Dynamic structure.  The ".dynamic" section contains an array of them.
type Dyn64 struct {
	Tag int64  /* Entry type. */
	Val uint64 /* Integer/address value */
}

/*
 * Relocation entries.
 */

/* ELF64 relocations that don't need an addend field. */
type Rel64 struct {
	Off  uint64 /* Location to be relocated. */
	Info uint64 /* Relocation type and symbol index. */
}

/* ELF64 relocations that need an addend field. */
type Rela64 struct {
	Off    uint64 /* Location to be relocated. */
	Info   uint64 /* Relocation type and symbol index. */
	Addend int64  /* Addend. */
}

func R_SYM64(info uint64) uint32    { return uint32(info >> 32) }
func R_TYPE64(info uint64) uint32   { return uint32(info) }
func R_INFO(sym, typ uint32) uint64 { return uint64(sym)<<32 | uint64(typ) }

// ELF64 symbol table entries.
type Sym64 struct {
	Name  uint32 /* String table index of name. */
	Info  uint8  /* Type and binding information. */
	Other uint8  /* Reserved (not used). */
	Shndx uint16 /* Section index of symbol. */
	Value uint64 /* Symbol value. */
	Size  uint64 /* Size of associated object. */
}

const Sym64Size = 24

type intName struct {
	i uint32
	s string
}

func stringName(i uint32, names []intName, goSyntax bool) string {
	for _, n := range names {
		if n.i == i {
			if goSyntax {
				return "elf." + n.s
			}
			return n.s
		}
	}

	// second pass - look for smaller to add with.
	// assume sorted already
	for j := len(names) - 1; j >= 0; j-- {
		n := names[j]
		if n.i < i {
			s := n.s
			if goSyntax {
				s = "elf." + s
			}
			return s + "+" + strconv.FormatUint(uint64(i-n.i), 10)
		}
	}

	return strconv.FormatUint(uint64(i), 10)
}

func flagName(i uint32, names []intName, goSyntax bool) string {
	s := ""
	for _, n := range names {
		if n.i&i == n.i {
			if len(s) > 0 {
				s += "+"
			}
			if goSyntax {
				s += "elf."
			}
			s += n.s
			i -= n.i
		}
	}
	if len(s) == 0 {
		return "0x" + strconv.FormatUint(uint64(i), 16)
	}
	if i != 0 {
		s += "+0x" + strconv.FormatUint(uint64(i), 16)
	}
	return s
}
