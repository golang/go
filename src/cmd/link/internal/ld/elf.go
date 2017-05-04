// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"cmd/internal/objabi"
	"cmd/internal/sys"
	"crypto/sha1"
	"encoding/binary"
	"encoding/hex"
	"io"
	"path/filepath"
	"sort"
	"strings"
)

/*
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
 * Portions Copyright 2009 The Go Authors. All rights reserved.
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
 *
 */

/*
 * ELF definitions that are independent of architecture or word size.
 */

/*
 * Note header.  The ".note" section contains an array of notes.  Each
 * begins with this header, aligned to a word boundary.  Immediately
 * following the note header is n_namesz bytes of name, padded to the
 * next word boundary.  Then comes n_descsz bytes of descriptor, again
 * padded to a word boundary.  The values of n_namesz and n_descsz do
 * not include the padding.
 */
type elfNote struct {
	nNamesz uint32
	nDescsz uint32
	nType   uint32
}

const (
	EI_MAG0              = 0
	EI_MAG1              = 1
	EI_MAG2              = 2
	EI_MAG3              = 3
	EI_CLASS             = 4
	EI_DATA              = 5
	EI_VERSION           = 6
	EI_OSABI             = 7
	EI_ABIVERSION        = 8
	OLD_EI_BRAND         = 8
	EI_PAD               = 9
	EI_NIDENT            = 16
	ELFMAG0              = 0x7f
	ELFMAG1              = 'E'
	ELFMAG2              = 'L'
	ELFMAG3              = 'F'
	SELFMAG              = 4
	EV_NONE              = 0
	EV_CURRENT           = 1
	ELFCLASSNONE         = 0
	ELFCLASS32           = 1
	ELFCLASS64           = 2
	ELFDATANONE          = 0
	ELFDATA2LSB          = 1
	ELFDATA2MSB          = 2
	ELFOSABI_NONE        = 0
	ELFOSABI_HPUX        = 1
	ELFOSABI_NETBSD      = 2
	ELFOSABI_LINUX       = 3
	ELFOSABI_HURD        = 4
	ELFOSABI_86OPEN      = 5
	ELFOSABI_SOLARIS     = 6
	ELFOSABI_AIX         = 7
	ELFOSABI_IRIX        = 8
	ELFOSABI_FREEBSD     = 9
	ELFOSABI_TRU64       = 10
	ELFOSABI_MODESTO     = 11
	ELFOSABI_OPENBSD     = 12
	ELFOSABI_OPENVMS     = 13
	ELFOSABI_NSK         = 14
	ELFOSABI_ARM         = 97
	ELFOSABI_STANDALONE  = 255
	ELFOSABI_SYSV        = ELFOSABI_NONE
	ELFOSABI_MONTEREY    = ELFOSABI_AIX
	ET_NONE              = 0
	ET_REL               = 1
	ET_EXEC              = 2
	ET_DYN               = 3
	ET_CORE              = 4
	ET_LOOS              = 0xfe00
	ET_HIOS              = 0xfeff
	ET_LOPROC            = 0xff00
	ET_HIPROC            = 0xffff
	EM_NONE              = 0
	EM_M32               = 1
	EM_SPARC             = 2
	EM_386               = 3
	EM_68K               = 4
	EM_88K               = 5
	EM_860               = 7
	EM_MIPS              = 8
	EM_S370              = 9
	EM_MIPS_RS3_LE       = 10
	EM_PARISC            = 15
	EM_VPP500            = 17
	EM_SPARC32PLUS       = 18
	EM_960               = 19
	EM_PPC               = 20
	EM_PPC64             = 21
	EM_S390              = 22
	EM_V800              = 36
	EM_FR20              = 37
	EM_RH32              = 38
	EM_RCE               = 39
	EM_ARM               = 40
	EM_SH                = 42
	EM_SPARCV9           = 43
	EM_TRICORE           = 44
	EM_ARC               = 45
	EM_H8_300            = 46
	EM_H8_300H           = 47
	EM_H8S               = 48
	EM_H8_500            = 49
	EM_IA_64             = 50
	EM_MIPS_X            = 51
	EM_COLDFIRE          = 52
	EM_68HC12            = 53
	EM_MMA               = 54
	EM_PCP               = 55
	EM_NCPU              = 56
	EM_NDR1              = 57
	EM_STARCORE          = 58
	EM_ME16              = 59
	EM_ST100             = 60
	EM_TINYJ             = 61
	EM_X86_64            = 62
	EM_AARCH64           = 183
	EM_486               = 6
	EM_MIPS_RS4_BE       = 10
	EM_ALPHA_STD         = 41
	EM_ALPHA             = 0x9026
	SHN_UNDEF            = 0
	SHN_LORESERVE        = 0xff00
	SHN_LOPROC           = 0xff00
	SHN_HIPROC           = 0xff1f
	SHN_LOOS             = 0xff20
	SHN_HIOS             = 0xff3f
	SHN_ABS              = 0xfff1
	SHN_COMMON           = 0xfff2
	SHN_XINDEX           = 0xffff
	SHN_HIRESERVE        = 0xffff
	SHT_NULL             = 0
	SHT_PROGBITS         = 1
	SHT_SYMTAB           = 2
	SHT_STRTAB           = 3
	SHT_RELA             = 4
	SHT_HASH             = 5
	SHT_DYNAMIC          = 6
	SHT_NOTE             = 7
	SHT_NOBITS           = 8
	SHT_REL              = 9
	SHT_SHLIB            = 10
	SHT_DYNSYM           = 11
	SHT_INIT_ARRAY       = 14
	SHT_FINI_ARRAY       = 15
	SHT_PREINIT_ARRAY    = 16
	SHT_GROUP            = 17
	SHT_SYMTAB_SHNDX     = 18
	SHT_LOOS             = 0x60000000
	SHT_HIOS             = 0x6fffffff
	SHT_GNU_VERDEF       = 0x6ffffffd
	SHT_GNU_VERNEED      = 0x6ffffffe
	SHT_GNU_VERSYM       = 0x6fffffff
	SHT_LOPROC           = 0x70000000
	SHT_ARM_ATTRIBUTES   = 0x70000003
	SHT_HIPROC           = 0x7fffffff
	SHT_LOUSER           = 0x80000000
	SHT_HIUSER           = 0xffffffff
	SHF_WRITE            = 0x1
	SHF_ALLOC            = 0x2
	SHF_EXECINSTR        = 0x4
	SHF_MERGE            = 0x10
	SHF_STRINGS          = 0x20
	SHF_INFO_LINK        = 0x40
	SHF_LINK_ORDER       = 0x80
	SHF_OS_NONCONFORMING = 0x100
	SHF_GROUP            = 0x200
	SHF_TLS              = 0x400
	SHF_MASKOS           = 0x0ff00000
	SHF_MASKPROC         = 0xf0000000
	PT_NULL              = 0
	PT_LOAD              = 1
	PT_DYNAMIC           = 2
	PT_INTERP            = 3
	PT_NOTE              = 4
	PT_SHLIB             = 5
	PT_PHDR              = 6
	PT_TLS               = 7
	PT_LOOS              = 0x60000000
	PT_HIOS              = 0x6fffffff
	PT_LOPROC            = 0x70000000
	PT_HIPROC            = 0x7fffffff
	PT_GNU_STACK         = 0x6474e551
	PT_GNU_RELRO         = 0x6474e552
	PT_PAX_FLAGS         = 0x65041580
	PT_SUNWSTACK         = 0x6ffffffb
	PF_X                 = 0x1
	PF_W                 = 0x2
	PF_R                 = 0x4
	PF_MASKOS            = 0x0ff00000
	PF_MASKPROC          = 0xf0000000
	DT_NULL              = 0
	DT_NEEDED            = 1
	DT_PLTRELSZ          = 2
	DT_PLTGOT            = 3
	DT_HASH              = 4
	DT_STRTAB            = 5
	DT_SYMTAB            = 6
	DT_RELA              = 7
	DT_RELASZ            = 8
	DT_RELAENT           = 9
	DT_STRSZ             = 10
	DT_SYMENT            = 11
	DT_INIT              = 12
	DT_FINI              = 13
	DT_SONAME            = 14
	DT_RPATH             = 15
	DT_SYMBOLIC          = 16
	DT_REL               = 17
	DT_RELSZ             = 18
	DT_RELENT            = 19
	DT_PLTREL            = 20
	DT_DEBUG             = 21
	DT_TEXTREL           = 22
	DT_JMPREL            = 23
	DT_BIND_NOW          = 24
	DT_INIT_ARRAY        = 25
	DT_FINI_ARRAY        = 26
	DT_INIT_ARRAYSZ      = 27
	DT_FINI_ARRAYSZ      = 28
	DT_RUNPATH           = 29
	DT_FLAGS             = 30
	DT_ENCODING          = 32
	DT_PREINIT_ARRAY     = 32
	DT_PREINIT_ARRAYSZ   = 33
	DT_LOOS              = 0x6000000d
	DT_HIOS              = 0x6ffff000
	DT_LOPROC            = 0x70000000
	DT_HIPROC            = 0x7fffffff
	DT_VERNEED           = 0x6ffffffe
	DT_VERNEEDNUM        = 0x6fffffff
	DT_VERSYM            = 0x6ffffff0
	DT_PPC64_GLINK       = DT_LOPROC + 0
	DT_PPC64_OPT         = DT_LOPROC + 3
	DF_ORIGIN            = 0x0001
	DF_SYMBOLIC          = 0x0002
	DF_TEXTREL           = 0x0004
	DF_BIND_NOW          = 0x0008
	DF_STATIC_TLS        = 0x0010
	NT_PRSTATUS          = 1
	NT_FPREGSET          = 2
	NT_PRPSINFO          = 3
	STB_LOCAL            = 0
	STB_GLOBAL           = 1
	STB_WEAK             = 2
	STB_LOOS             = 10
	STB_HIOS             = 12
	STB_LOPROC           = 13
	STB_HIPROC           = 15
	STT_NOTYPE           = 0
	STT_OBJECT           = 1
	STT_FUNC             = 2
	STT_SECTION          = 3
	STT_FILE             = 4
	STT_COMMON           = 5
	STT_TLS              = 6
	STT_LOOS             = 10
	STT_HIOS             = 12
	STT_LOPROC           = 13
	STT_HIPROC           = 15
	STV_DEFAULT          = 0x0
	STV_INTERNAL         = 0x1
	STV_HIDDEN           = 0x2
	STV_PROTECTED        = 0x3
	STN_UNDEF            = 0
)

/* For accessing the fields of r_info. */

/* For constructing r_info from field values. */

/*
 * Relocation types.
 */
const (
	R_X86_64_NONE           = 0
	R_X86_64_64             = 1
	R_X86_64_PC32           = 2
	R_X86_64_GOT32          = 3
	R_X86_64_PLT32          = 4
	R_X86_64_COPY           = 5
	R_X86_64_GLOB_DAT       = 6
	R_X86_64_JMP_SLOT       = 7
	R_X86_64_RELATIVE       = 8
	R_X86_64_GOTPCREL       = 9
	R_X86_64_32             = 10
	R_X86_64_32S            = 11
	R_X86_64_16             = 12
	R_X86_64_PC16           = 13
	R_X86_64_8              = 14
	R_X86_64_PC8            = 15
	R_X86_64_DTPMOD64       = 16
	R_X86_64_DTPOFF64       = 17
	R_X86_64_TPOFF64        = 18
	R_X86_64_TLSGD          = 19
	R_X86_64_TLSLD          = 20
	R_X86_64_DTPOFF32       = 21
	R_X86_64_GOTTPOFF       = 22
	R_X86_64_TPOFF32        = 23
	R_X86_64_PC64           = 24
	R_X86_64_GOTOFF64       = 25
	R_X86_64_GOTPC32        = 26
	R_X86_64_GOT64          = 27
	R_X86_64_GOTPCREL64     = 28
	R_X86_64_GOTPC64        = 29
	R_X86_64_GOTPLT64       = 30
	R_X86_64_PLTOFF64       = 31
	R_X86_64_SIZE32         = 32
	R_X86_64_SIZE64         = 33
	R_X86_64_GOTPC32_TLSDEC = 34
	R_X86_64_TLSDESC_CALL   = 35
	R_X86_64_TLSDESC        = 36
	R_X86_64_IRELATIVE      = 37
	R_X86_64_PC32_BND       = 40
	R_X86_64_GOTPCRELX      = 41
	R_X86_64_REX_GOTPCRELX  = 42

	R_AARCH64_ABS64                       = 257
	R_AARCH64_ABS32                       = 258
	R_AARCH64_CALL26                      = 283
	R_AARCH64_ADR_PREL_PG_HI21            = 275
	R_AARCH64_ADD_ABS_LO12_NC             = 277
	R_AARCH64_LDST8_ABS_LO12_NC           = 278
	R_AARCH64_LDST16_ABS_LO12_NC          = 284
	R_AARCH64_LDST32_ABS_LO12_NC          = 285
	R_AARCH64_LDST64_ABS_LO12_NC          = 286
	R_AARCH64_ADR_GOT_PAGE                = 311
	R_AARCH64_LD64_GOT_LO12_NC            = 312
	R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21   = 541
	R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC = 542
	R_AARCH64_TLSLE_MOVW_TPREL_G0         = 547

	R_ALPHA_NONE           = 0
	R_ALPHA_REFLONG        = 1
	R_ALPHA_REFQUAD        = 2
	R_ALPHA_GPREL32        = 3
	R_ALPHA_LITERAL        = 4
	R_ALPHA_LITUSE         = 5
	R_ALPHA_GPDISP         = 6
	R_ALPHA_BRADDR         = 7
	R_ALPHA_HINT           = 8
	R_ALPHA_SREL16         = 9
	R_ALPHA_SREL32         = 10
	R_ALPHA_SREL64         = 11
	R_ALPHA_OP_PUSH        = 12
	R_ALPHA_OP_STORE       = 13
	R_ALPHA_OP_PSUB        = 14
	R_ALPHA_OP_PRSHIFT     = 15
	R_ALPHA_GPVALUE        = 16
	R_ALPHA_GPRELHIGH      = 17
	R_ALPHA_GPRELLOW       = 18
	R_ALPHA_IMMED_GP_16    = 19
	R_ALPHA_IMMED_GP_HI32  = 20
	R_ALPHA_IMMED_SCN_HI32 = 21
	R_ALPHA_IMMED_BR_HI32  = 22
	R_ALPHA_IMMED_LO32     = 23
	R_ALPHA_COPY           = 24
	R_ALPHA_GLOB_DAT       = 25
	R_ALPHA_JMP_SLOT       = 26
	R_ALPHA_RELATIVE       = 27

	R_ARM_NONE          = 0
	R_ARM_PC24          = 1
	R_ARM_ABS32         = 2
	R_ARM_REL32         = 3
	R_ARM_PC13          = 4
	R_ARM_ABS16         = 5
	R_ARM_ABS12         = 6
	R_ARM_THM_ABS5      = 7
	R_ARM_ABS8          = 8
	R_ARM_SBREL32       = 9
	R_ARM_THM_PC22      = 10
	R_ARM_THM_PC8       = 11
	R_ARM_AMP_VCALL9    = 12
	R_ARM_SWI24         = 13
	R_ARM_THM_SWI8      = 14
	R_ARM_XPC25         = 15
	R_ARM_THM_XPC22     = 16
	R_ARM_COPY          = 20
	R_ARM_GLOB_DAT      = 21
	R_ARM_JUMP_SLOT     = 22
	R_ARM_RELATIVE      = 23
	R_ARM_GOTOFF        = 24
	R_ARM_GOTPC         = 25
	R_ARM_GOT32         = 26
	R_ARM_PLT32         = 27
	R_ARM_CALL          = 28
	R_ARM_JUMP24        = 29
	R_ARM_V4BX          = 40
	R_ARM_GOT_PREL      = 96
	R_ARM_GNU_VTENTRY   = 100
	R_ARM_GNU_VTINHERIT = 101
	R_ARM_TLS_IE32      = 107
	R_ARM_TLS_LE32      = 108
	R_ARM_RSBREL32      = 250
	R_ARM_THM_RPC22     = 251
	R_ARM_RREL32        = 252
	R_ARM_RABS32        = 253
	R_ARM_RPC24         = 254
	R_ARM_RBASE         = 255

	R_386_NONE          = 0
	R_386_32            = 1
	R_386_PC32          = 2
	R_386_GOT32         = 3
	R_386_PLT32         = 4
	R_386_COPY          = 5
	R_386_GLOB_DAT      = 6
	R_386_JMP_SLOT      = 7
	R_386_RELATIVE      = 8
	R_386_GOTOFF        = 9
	R_386_GOTPC         = 10
	R_386_TLS_TPOFF     = 14
	R_386_TLS_IE        = 15
	R_386_TLS_GOTIE     = 16
	R_386_TLS_LE        = 17
	R_386_TLS_GD        = 18
	R_386_TLS_LDM       = 19
	R_386_TLS_GD_32     = 24
	R_386_TLS_GD_PUSH   = 25
	R_386_TLS_GD_CALL   = 26
	R_386_TLS_GD_POP    = 27
	R_386_TLS_LDM_32    = 28
	R_386_TLS_LDM_PUSH  = 29
	R_386_TLS_LDM_CALL  = 30
	R_386_TLS_LDM_POP   = 31
	R_386_TLS_LDO_32    = 32
	R_386_TLS_IE_32     = 33
	R_386_TLS_LE_32     = 34
	R_386_TLS_DTPMOD32  = 35
	R_386_TLS_DTPOFF32  = 36
	R_386_TLS_TPOFF32   = 37
	R_386_TLS_GOTDESC   = 39
	R_386_TLS_DESC_CALL = 40
	R_386_TLS_DESC      = 41
	R_386_IRELATIVE     = 42
	R_386_GOT32X        = 43

	R_MIPS_NONE            = 0
	R_MIPS_16              = 1
	R_MIPS_32              = 2
	R_MIPS_REL32           = 3
	R_MIPS_26              = 4
	R_MIPS_HI16            = 5
	R_MIPS_LO16            = 6
	R_MIPS_GPREL16         = 7
	R_MIPS_LITERAL         = 8
	R_MIPS_GOT16           = 9
	R_MIPS_PC16            = 10
	R_MIPS_CALL16          = 11
	R_MIPS_GPREL32         = 12
	R_MIPS_SHIFT5          = 16
	R_MIPS_SHIFT6          = 17
	R_MIPS_64              = 18
	R_MIPS_GOT_DISP        = 19
	R_MIPS_GOT_PAGE        = 20
	R_MIPS_GOT_OFST        = 21
	R_MIPS_GOT_HI16        = 22
	R_MIPS_GOT_LO16        = 23
	R_MIPS_SUB             = 24
	R_MIPS_INSERT_A        = 25
	R_MIPS_INSERT_B        = 26
	R_MIPS_DELETE          = 27
	R_MIPS_HIGHER          = 28
	R_MIPS_HIGHEST         = 29
	R_MIPS_CALL_HI16       = 30
	R_MIPS_CALL_LO16       = 31
	R_MIPS_SCN_DISP        = 32
	R_MIPS_REL16           = 33
	R_MIPS_ADD_IMMEDIATE   = 34
	R_MIPS_PJUMP           = 35
	R_MIPS_RELGOT          = 36
	R_MIPS_JALR            = 37
	R_MIPS_TLS_DTPMOD32    = 38
	R_MIPS_TLS_DTPREL32    = 39
	R_MIPS_TLS_DTPMOD64    = 40
	R_MIPS_TLS_DTPREL64    = 41
	R_MIPS_TLS_GD          = 42
	R_MIPS_TLS_LDM         = 43
	R_MIPS_TLS_DTPREL_HI16 = 44
	R_MIPS_TLS_DTPREL_LO16 = 45
	R_MIPS_TLS_GOTTPREL    = 46
	R_MIPS_TLS_TPREL32     = 47
	R_MIPS_TLS_TPREL64     = 48
	R_MIPS_TLS_TPREL_HI16  = 49
	R_MIPS_TLS_TPREL_LO16  = 50

	R_PPC_NONE            = 0
	R_PPC_ADDR32          = 1
	R_PPC_ADDR24          = 2
	R_PPC_ADDR16          = 3
	R_PPC_ADDR16_LO       = 4
	R_PPC_ADDR16_HI       = 5
	R_PPC_ADDR16_HA       = 6
	R_PPC_ADDR14          = 7
	R_PPC_ADDR14_BRTAKEN  = 8
	R_PPC_ADDR14_BRNTAKEN = 9
	R_PPC_REL24           = 10
	R_PPC_REL14           = 11
	R_PPC_REL14_BRTAKEN   = 12
	R_PPC_REL14_BRNTAKEN  = 13
	R_PPC_GOT16           = 14
	R_PPC_GOT16_LO        = 15
	R_PPC_GOT16_HI        = 16
	R_PPC_GOT16_HA        = 17
	R_PPC_PLTREL24        = 18
	R_PPC_COPY            = 19
	R_PPC_GLOB_DAT        = 20
	R_PPC_JMP_SLOT        = 21
	R_PPC_RELATIVE        = 22
	R_PPC_LOCAL24PC       = 23
	R_PPC_UADDR32         = 24
	R_PPC_UADDR16         = 25
	R_PPC_REL32           = 26
	R_PPC_PLT32           = 27
	R_PPC_PLTREL32        = 28
	R_PPC_PLT16_LO        = 29
	R_PPC_PLT16_HI        = 30
	R_PPC_PLT16_HA        = 31
	R_PPC_SDAREL16        = 32
	R_PPC_SECTOFF         = 33
	R_PPC_SECTOFF_LO      = 34
	R_PPC_SECTOFF_HI      = 35
	R_PPC_SECTOFF_HA      = 36
	R_PPC_TLS             = 67
	R_PPC_DTPMOD32        = 68
	R_PPC_TPREL16         = 69
	R_PPC_TPREL16_LO      = 70
	R_PPC_TPREL16_HI      = 71
	R_PPC_TPREL16_HA      = 72
	R_PPC_TPREL32         = 73
	R_PPC_DTPREL16        = 74
	R_PPC_DTPREL16_LO     = 75
	R_PPC_DTPREL16_HI     = 76
	R_PPC_DTPREL16_HA     = 77
	R_PPC_DTPREL32        = 78
	R_PPC_GOT_TLSGD16     = 79
	R_PPC_GOT_TLSGD16_LO  = 80
	R_PPC_GOT_TLSGD16_HI  = 81
	R_PPC_GOT_TLSGD16_HA  = 82
	R_PPC_GOT_TLSLD16     = 83
	R_PPC_GOT_TLSLD16_LO  = 84
	R_PPC_GOT_TLSLD16_HI  = 85
	R_PPC_GOT_TLSLD16_HA  = 86
	R_PPC_GOT_TPREL16     = 87
	R_PPC_GOT_TPREL16_LO  = 88
	R_PPC_GOT_TPREL16_HI  = 89
	R_PPC_GOT_TPREL16_HA  = 90
	R_PPC_EMB_NADDR32     = 101
	R_PPC_EMB_NADDR16     = 102
	R_PPC_EMB_NADDR16_LO  = 103
	R_PPC_EMB_NADDR16_HI  = 104
	R_PPC_EMB_NADDR16_HA  = 105
	R_PPC_EMB_SDAI16      = 106
	R_PPC_EMB_SDA2I16     = 107
	R_PPC_EMB_SDA2REL     = 108
	R_PPC_EMB_SDA21       = 109
	R_PPC_EMB_MRKREF      = 110
	R_PPC_EMB_RELSEC16    = 111
	R_PPC_EMB_RELST_LO    = 112
	R_PPC_EMB_RELST_HI    = 113
	R_PPC_EMB_RELST_HA    = 114
	R_PPC_EMB_BIT_FLD     = 115
	R_PPC_EMB_RELSDA      = 116

	R_PPC64_ADDR32            = R_PPC_ADDR32
	R_PPC64_ADDR16_LO         = R_PPC_ADDR16_LO
	R_PPC64_ADDR16_HA         = R_PPC_ADDR16_HA
	R_PPC64_REL24             = R_PPC_REL24
	R_PPC64_GOT16_HA          = R_PPC_GOT16_HA
	R_PPC64_JMP_SLOT          = R_PPC_JMP_SLOT
	R_PPC64_TPREL16           = R_PPC_TPREL16
	R_PPC64_ADDR64            = 38
	R_PPC64_TOC16             = 47
	R_PPC64_TOC16_LO          = 48
	R_PPC64_TOC16_HI          = 49
	R_PPC64_TOC16_HA          = 50
	R_PPC64_ADDR16_LO_DS      = 57
	R_PPC64_GOT16_LO_DS       = 59
	R_PPC64_TOC16_DS          = 63
	R_PPC64_TOC16_LO_DS       = 64
	R_PPC64_TLS               = 67
	R_PPC64_GOT_TPREL16_LO_DS = 88
	R_PPC64_GOT_TPREL16_HA    = 90
	R_PPC64_REL16_LO          = 250
	R_PPC64_REL16_HI          = 251
	R_PPC64_REL16_HA          = 252

	R_SPARC_NONE     = 0
	R_SPARC_8        = 1
	R_SPARC_16       = 2
	R_SPARC_32       = 3
	R_SPARC_DISP8    = 4
	R_SPARC_DISP16   = 5
	R_SPARC_DISP32   = 6
	R_SPARC_WDISP30  = 7
	R_SPARC_WDISP22  = 8
	R_SPARC_HI22     = 9
	R_SPARC_22       = 10
	R_SPARC_13       = 11
	R_SPARC_LO10     = 12
	R_SPARC_GOT10    = 13
	R_SPARC_GOT13    = 14
	R_SPARC_GOT22    = 15
	R_SPARC_PC10     = 16
	R_SPARC_PC22     = 17
	R_SPARC_WPLT30   = 18
	R_SPARC_COPY     = 19
	R_SPARC_GLOB_DAT = 20
	R_SPARC_JMP_SLOT = 21
	R_SPARC_RELATIVE = 22
	R_SPARC_UA32     = 23
	R_SPARC_PLT32    = 24
	R_SPARC_HIPLT22  = 25
	R_SPARC_LOPLT10  = 26
	R_SPARC_PCPLT32  = 27
	R_SPARC_PCPLT22  = 28
	R_SPARC_PCPLT10  = 29
	R_SPARC_10       = 30
	R_SPARC_11       = 31
	R_SPARC_64       = 32
	R_SPARC_OLO10    = 33
	R_SPARC_HH22     = 34
	R_SPARC_HM10     = 35
	R_SPARC_LM22     = 36
	R_SPARC_PC_HH22  = 37
	R_SPARC_PC_HM10  = 38
	R_SPARC_PC_LM22  = 39
	R_SPARC_WDISP16  = 40
	R_SPARC_WDISP19  = 41
	R_SPARC_GLOB_JMP = 42
	R_SPARC_7        = 43
	R_SPARC_5        = 44
	R_SPARC_6        = 45
	R_SPARC_DISP64   = 46
	R_SPARC_PLT64    = 47
	R_SPARC_HIX22    = 48
	R_SPARC_LOX10    = 49
	R_SPARC_H44      = 50
	R_SPARC_M44      = 51
	R_SPARC_L44      = 52
	R_SPARC_REGISTER = 53
	R_SPARC_UA64     = 54
	R_SPARC_UA16     = 55

	R_390_NONE        = 0
	R_390_8           = 1
	R_390_12          = 2
	R_390_16          = 3
	R_390_32          = 4
	R_390_PC32        = 5
	R_390_GOT12       = 6
	R_390_GOT32       = 7
	R_390_PLT32       = 8
	R_390_COPY        = 9
	R_390_GLOB_DAT    = 10
	R_390_JMP_SLOT    = 11
	R_390_RELATIVE    = 12
	R_390_GOTOFF      = 13
	R_390_GOTPC       = 14
	R_390_GOT16       = 15
	R_390_PC16        = 16
	R_390_PC16DBL     = 17
	R_390_PLT16DBL    = 18
	R_390_PC32DBL     = 19
	R_390_PLT32DBL    = 20
	R_390_GOTPCDBL    = 21
	R_390_64          = 22
	R_390_PC64        = 23
	R_390_GOT64       = 24
	R_390_PLT64       = 25
	R_390_GOTENT      = 26
	R_390_GOTOFF16    = 27
	R_390_GOTOFF64    = 28
	R_390_GOTPLT12    = 29
	R_390_GOTPLT16    = 30
	R_390_GOTPLT32    = 31
	R_390_GOTPLT64    = 32
	R_390_GOTPLTENT   = 33
	R_390_GOTPLTOFF16 = 34
	R_390_GOTPLTOFF32 = 35
	R_390_GOTPLTOFF64 = 36
	R_390_TLS_LOAD    = 37
	R_390_TLS_GDCALL  = 38
	R_390_TLS_LDCALL  = 39
	R_390_TLS_GD32    = 40
	R_390_TLS_GD64    = 41
	R_390_TLS_GOTIE12 = 42
	R_390_TLS_GOTIE32 = 43
	R_390_TLS_GOTIE64 = 44
	R_390_TLS_LDM32   = 45
	R_390_TLS_LDM64   = 46
	R_390_TLS_IE32    = 47
	R_390_TLS_IE64    = 48
	R_390_TLS_IEENT   = 49
	R_390_TLS_LE32    = 50
	R_390_TLS_LE64    = 51
	R_390_TLS_LDO32   = 52
	R_390_TLS_LDO64   = 53
	R_390_TLS_DTPMOD  = 54
	R_390_TLS_DTPOFF  = 55
	R_390_TLS_TPOFF   = 56
	R_390_20          = 57
	R_390_GOT20       = 58
	R_390_GOTPLT20    = 59
	R_390_TLS_GOTIE20 = 60

	ARM_MAGIC_TRAMP_NUMBER = 0x5c000003
)

/*
 * Symbol table entries.
 */

/* For accessing the fields of st_info. */

/* For constructing st_info from field values. */

/* For accessing the fields of st_other. */

/*
 * ELF header.
 */
type ElfEhdr struct {
	ident     [EI_NIDENT]uint8
	type_     uint16
	machine   uint16
	version   uint32
	entry     uint64
	phoff     uint64
	shoff     uint64
	flags     uint32
	ehsize    uint16
	phentsize uint16
	phnum     uint16
	shentsize uint16
	shnum     uint16
	shstrndx  uint16
}

/*
 * Section header.
 */
type ElfShdr struct {
	name      uint32
	type_     uint32
	flags     uint64
	addr      uint64
	off       uint64
	size      uint64
	link      uint32
	info      uint32
	addralign uint64
	entsize   uint64
	shnum     int
	secsym    *Symbol
}

/*
 * Program header.
 */
type ElfPhdr struct {
	type_  uint32
	flags  uint32
	off    uint64
	vaddr  uint64
	paddr  uint64
	filesz uint64
	memsz  uint64
	align  uint64
}

/* For accessing the fields of r_info. */

/* For constructing r_info from field values. */

/*
 * Symbol table entries.
 */

/* For accessing the fields of st_info. */

/* For constructing st_info from field values. */

/* For accessing the fields of st_other. */

/*
 * Go linker interface
 */
const (
	ELF64HDRSIZE  = 64
	ELF64PHDRSIZE = 56
	ELF64SHDRSIZE = 64
	ELF64RELSIZE  = 16
	ELF64RELASIZE = 24
	ELF64SYMSIZE  = 24
	ELF32HDRSIZE  = 52
	ELF32PHDRSIZE = 32
	ELF32SHDRSIZE = 40
	ELF32SYMSIZE  = 16
	ELF32RELSIZE  = 8
)

/*
 * The interface uses the 64-bit structures always,
 * to avoid code duplication.  The writers know how to
 * marshal a 32-bit representation from the 64-bit structure.
 */

var Elfstrdat []byte

/*
 * Total amount of space to reserve at the start of the file
 * for Header, PHeaders, SHeaders, and interp.
 * May waste some.
 * On FreeBSD, cannot be larger than a page.
 */
const (
	ELFRESERVE = 4096
)

/*
 * We use the 64-bit data structures on both 32- and 64-bit machines
 * in order to write the code just once.  The 64-bit data structure is
 * written in the 32-bit format on the 32-bit machines.
 */
const (
	NSECT = 400
)

var (
	Iself bool

	Nelfsym int = 1

	elf64 bool
	// Either ".rel" or ".rela" depending on which type of relocation the
	// target platform uses.
	elfRelType string

	ehdr ElfEhdr
	phdr [NSECT]*ElfPhdr
	shdr [NSECT]*ElfShdr

	interp string
)

type Elfstring struct {
	s   string
	off int
}

var elfstr [100]Elfstring

var nelfstr int

var buildinfo []byte

/*
 Initialize the global variable that describes the ELF header. It will be updated as
 we write section and prog headers.
*/
func Elfinit(ctxt *Link) {
	Iself = true

	if SysArch.InFamily(sys.AMD64, sys.ARM64, sys.MIPS64, sys.PPC64, sys.S390X) {
		elfRelType = ".rela"
	} else {
		elfRelType = ".rel"
	}

	switch SysArch.Family {
	// 64-bit architectures
	case sys.PPC64, sys.S390X:
		if ctxt.Arch.ByteOrder == binary.BigEndian {
			ehdr.flags = 1 /* Version 1 ABI */
		} else {
			ehdr.flags = 2 /* Version 2 ABI */
		}
		fallthrough
	case sys.AMD64, sys.ARM64, sys.MIPS64:
		if SysArch.Family == sys.MIPS64 {
			ehdr.flags = 0x20000004 /* MIPS 3 CPIC */
		}
		elf64 = true

		ehdr.phoff = ELF64HDRSIZE      /* Must be be ELF64HDRSIZE: first PHdr must follow ELF header */
		ehdr.shoff = ELF64HDRSIZE      /* Will move as we add PHeaders */
		ehdr.ehsize = ELF64HDRSIZE     /* Must be ELF64HDRSIZE */
		ehdr.phentsize = ELF64PHDRSIZE /* Must be ELF64PHDRSIZE */
		ehdr.shentsize = ELF64SHDRSIZE /* Must be ELF64SHDRSIZE */

	// 32-bit architectures
	case sys.ARM, sys.MIPS:
		if SysArch.Family == sys.ARM {
			// we use EABI on linux/arm, freebsd/arm, netbsd/arm.
			if Headtype == objabi.Hlinux || Headtype == objabi.Hfreebsd || Headtype == objabi.Hnetbsd {
				// We set a value here that makes no indication of which
				// float ABI the object uses, because this is information
				// used by the dynamic linker to compare executables and
				// shared libraries -- so it only matters for cgo calls, and
				// the information properly comes from the object files
				// produced by the host C compiler. parseArmAttributes in
				// ldelf.go reads that information and updates this field as
				// appropriate.
				ehdr.flags = 0x5000002 // has entry point, Version5 EABI
			}
		} else if SysArch.Family == sys.MIPS {
			ehdr.flags = 0x50001004 /* MIPS 32 CPIC O32*/
		}
		fallthrough
	default:
		ehdr.phoff = ELF32HDRSIZE
		/* Must be be ELF32HDRSIZE: first PHdr must follow ELF header */
		ehdr.shoff = ELF32HDRSIZE      /* Will move as we add PHeaders */
		ehdr.ehsize = ELF32HDRSIZE     /* Must be ELF32HDRSIZE */
		ehdr.phentsize = ELF32PHDRSIZE /* Must be ELF32PHDRSIZE */
		ehdr.shentsize = ELF32SHDRSIZE /* Must be ELF32SHDRSIZE */
	}
}

// Make sure PT_LOAD is aligned properly and
// that there is no gap,
// correct ELF loaders will do this implicitly,
// but buggy ELF loaders like the one in some
// versions of QEMU and UPX won't.
func fixElfPhdr(e *ElfPhdr) {
	frag := int(e.vaddr & (e.align - 1))

	e.off -= uint64(frag)
	e.vaddr -= uint64(frag)
	e.paddr -= uint64(frag)
	e.filesz += uint64(frag)
	e.memsz += uint64(frag)
}

func elf64phdr(e *ElfPhdr) {
	if e.type_ == PT_LOAD {
		fixElfPhdr(e)
	}

	Thearch.Lput(e.type_)
	Thearch.Lput(e.flags)
	Thearch.Vput(e.off)
	Thearch.Vput(e.vaddr)
	Thearch.Vput(e.paddr)
	Thearch.Vput(e.filesz)
	Thearch.Vput(e.memsz)
	Thearch.Vput(e.align)
}

func elf32phdr(e *ElfPhdr) {
	if e.type_ == PT_LOAD {
		fixElfPhdr(e)
	}

	Thearch.Lput(e.type_)
	Thearch.Lput(uint32(e.off))
	Thearch.Lput(uint32(e.vaddr))
	Thearch.Lput(uint32(e.paddr))
	Thearch.Lput(uint32(e.filesz))
	Thearch.Lput(uint32(e.memsz))
	Thearch.Lput(e.flags)
	Thearch.Lput(uint32(e.align))
}

func elf64shdr(e *ElfShdr) {
	Thearch.Lput(e.name)
	Thearch.Lput(e.type_)
	Thearch.Vput(e.flags)
	Thearch.Vput(e.addr)
	Thearch.Vput(e.off)
	Thearch.Vput(e.size)
	Thearch.Lput(e.link)
	Thearch.Lput(e.info)
	Thearch.Vput(e.addralign)
	Thearch.Vput(e.entsize)
}

func elf32shdr(e *ElfShdr) {
	Thearch.Lput(e.name)
	Thearch.Lput(e.type_)
	Thearch.Lput(uint32(e.flags))
	Thearch.Lput(uint32(e.addr))
	Thearch.Lput(uint32(e.off))
	Thearch.Lput(uint32(e.size))
	Thearch.Lput(e.link)
	Thearch.Lput(e.info)
	Thearch.Lput(uint32(e.addralign))
	Thearch.Lput(uint32(e.entsize))
}

func elfwriteshdrs() uint32 {
	if elf64 {
		for i := 0; i < int(ehdr.shnum); i++ {
			elf64shdr(shdr[i])
		}
		return uint32(ehdr.shnum) * ELF64SHDRSIZE
	}

	for i := 0; i < int(ehdr.shnum); i++ {
		elf32shdr(shdr[i])
	}
	return uint32(ehdr.shnum) * ELF32SHDRSIZE
}

func elfsetstring(s *Symbol, str string, off int) {
	if nelfstr >= len(elfstr) {
		Errorf(s, "too many elf strings")
		errorexit()
	}

	elfstr[nelfstr].s = str
	elfstr[nelfstr].off = off
	nelfstr++
}

func elfwritephdrs() uint32 {
	if elf64 {
		for i := 0; i < int(ehdr.phnum); i++ {
			elf64phdr(phdr[i])
		}
		return uint32(ehdr.phnum) * ELF64PHDRSIZE
	}

	for i := 0; i < int(ehdr.phnum); i++ {
		elf32phdr(phdr[i])
	}
	return uint32(ehdr.phnum) * ELF32PHDRSIZE
}

func newElfPhdr() *ElfPhdr {
	e := new(ElfPhdr)
	if ehdr.phnum >= NSECT {
		Errorf(nil, "too many phdrs")
	} else {
		phdr[ehdr.phnum] = e
		ehdr.phnum++
	}
	if elf64 {
		ehdr.shoff += ELF64PHDRSIZE
	} else {
		ehdr.shoff += ELF32PHDRSIZE
	}
	return e
}

func newElfShdr(name int64) *ElfShdr {
	e := new(ElfShdr)
	e.name = uint32(name)
	e.shnum = int(ehdr.shnum)
	if ehdr.shnum >= NSECT {
		Errorf(nil, "too many shdrs")
	} else {
		shdr[ehdr.shnum] = e
		ehdr.shnum++
	}

	return e
}

func getElfEhdr() *ElfEhdr {
	return &ehdr
}

func elf64writehdr() uint32 {
	for i := 0; i < EI_NIDENT; i++ {
		Cput(ehdr.ident[i])
	}
	Thearch.Wput(ehdr.type_)
	Thearch.Wput(ehdr.machine)
	Thearch.Lput(ehdr.version)
	Thearch.Vput(ehdr.entry)
	Thearch.Vput(ehdr.phoff)
	Thearch.Vput(ehdr.shoff)
	Thearch.Lput(ehdr.flags)
	Thearch.Wput(ehdr.ehsize)
	Thearch.Wput(ehdr.phentsize)
	Thearch.Wput(ehdr.phnum)
	Thearch.Wput(ehdr.shentsize)
	Thearch.Wput(ehdr.shnum)
	Thearch.Wput(ehdr.shstrndx)
	return ELF64HDRSIZE
}

func elf32writehdr() uint32 {
	for i := 0; i < EI_NIDENT; i++ {
		Cput(ehdr.ident[i])
	}
	Thearch.Wput(ehdr.type_)
	Thearch.Wput(ehdr.machine)
	Thearch.Lput(ehdr.version)
	Thearch.Lput(uint32(ehdr.entry))
	Thearch.Lput(uint32(ehdr.phoff))
	Thearch.Lput(uint32(ehdr.shoff))
	Thearch.Lput(ehdr.flags)
	Thearch.Wput(ehdr.ehsize)
	Thearch.Wput(ehdr.phentsize)
	Thearch.Wput(ehdr.phnum)
	Thearch.Wput(ehdr.shentsize)
	Thearch.Wput(ehdr.shnum)
	Thearch.Wput(ehdr.shstrndx)
	return ELF32HDRSIZE
}

func elfwritehdr() uint32 {
	if elf64 {
		return elf64writehdr()
	}
	return elf32writehdr()
}

/* Taken directly from the definition document for ELF64 */
func elfhash(name string) uint32 {
	var h uint32
	for i := 0; i < len(name); i++ {
		h = (h << 4) + uint32(name[i])
		if g := h & 0xf0000000; g != 0 {
			h ^= g >> 24
		}
		h &= 0x0fffffff
	}
	return h
}

func Elfwritedynent(ctxt *Link, s *Symbol, tag int, val uint64) {
	if elf64 {
		Adduint64(ctxt, s, uint64(tag))
		Adduint64(ctxt, s, val)
	} else {
		Adduint32(ctxt, s, uint32(tag))
		Adduint32(ctxt, s, uint32(val))
	}
}

func elfwritedynentsym(ctxt *Link, s *Symbol, tag int, t *Symbol) {
	Elfwritedynentsymplus(ctxt, s, tag, t, 0)
}

func Elfwritedynentsymplus(ctxt *Link, s *Symbol, tag int, t *Symbol, add int64) {
	if elf64 {
		Adduint64(ctxt, s, uint64(tag))
	} else {
		Adduint32(ctxt, s, uint32(tag))
	}
	Addaddrplus(ctxt, s, t, add)
}

func elfwritedynentsymsize(ctxt *Link, s *Symbol, tag int, t *Symbol) {
	if elf64 {
		Adduint64(ctxt, s, uint64(tag))
	} else {
		Adduint32(ctxt, s, uint32(tag))
	}
	addsize(ctxt, s, t)
}

func elfinterp(sh *ElfShdr, startva uint64, resoff uint64, p string) int {
	interp = p
	n := len(interp) + 1
	sh.addr = startva + resoff - uint64(n)
	sh.off = resoff - uint64(n)
	sh.size = uint64(n)

	return n
}

func elfwriteinterp() int {
	sh := elfshname(".interp")
	Cseek(int64(sh.off))
	coutbuf.WriteString(interp)
	Cput(0)
	return int(sh.size)
}

func elfnote(sh *ElfShdr, startva uint64, resoff uint64, sz int, alloc bool) int {
	n := 3*4 + uint64(sz) + resoff%4

	sh.type_ = SHT_NOTE
	if alloc {
		sh.flags = SHF_ALLOC
	}
	sh.addralign = 4
	sh.addr = startva + resoff - n
	sh.off = resoff - n
	sh.size = n - resoff%4

	return int(n)
}

func elfwritenotehdr(str string, namesz uint32, descsz uint32, tag uint32) *ElfShdr {
	sh := elfshname(str)

	// Write Elf_Note header.
	Cseek(int64(sh.off))

	Thearch.Lput(namesz)
	Thearch.Lput(descsz)
	Thearch.Lput(tag)

	return sh
}

// NetBSD Signature (as per sys/exec_elf.h)
const (
	ELF_NOTE_NETBSD_NAMESZ  = 7
	ELF_NOTE_NETBSD_DESCSZ  = 4
	ELF_NOTE_NETBSD_TAG     = 1
	ELF_NOTE_NETBSD_VERSION = 599000000 /* NetBSD 5.99 */
)

var ELF_NOTE_NETBSD_NAME = []byte("NetBSD\x00")

func elfnetbsdsig(sh *ElfShdr, startva uint64, resoff uint64) int {
	n := int(Rnd(ELF_NOTE_NETBSD_NAMESZ, 4) + Rnd(ELF_NOTE_NETBSD_DESCSZ, 4))
	return elfnote(sh, startva, resoff, n, true)
}

func elfwritenetbsdsig() int {
	// Write Elf_Note header.
	sh := elfwritenotehdr(".note.netbsd.ident", ELF_NOTE_NETBSD_NAMESZ, ELF_NOTE_NETBSD_DESCSZ, ELF_NOTE_NETBSD_TAG)

	if sh == nil {
		return 0
	}

	// Followed by NetBSD string and version.
	Cwrite(ELF_NOTE_NETBSD_NAME)
	Cput(0)

	Thearch.Lput(ELF_NOTE_NETBSD_VERSION)

	return int(sh.size)
}

// OpenBSD Signature
const (
	ELF_NOTE_OPENBSD_NAMESZ  = 8
	ELF_NOTE_OPENBSD_DESCSZ  = 4
	ELF_NOTE_OPENBSD_TAG     = 1
	ELF_NOTE_OPENBSD_VERSION = 0
)

var ELF_NOTE_OPENBSD_NAME = []byte("OpenBSD\x00")

func elfopenbsdsig(sh *ElfShdr, startva uint64, resoff uint64) int {
	n := ELF_NOTE_OPENBSD_NAMESZ + ELF_NOTE_OPENBSD_DESCSZ
	return elfnote(sh, startva, resoff, n, true)
}

func elfwriteopenbsdsig() int {
	// Write Elf_Note header.
	sh := elfwritenotehdr(".note.openbsd.ident", ELF_NOTE_OPENBSD_NAMESZ, ELF_NOTE_OPENBSD_DESCSZ, ELF_NOTE_OPENBSD_TAG)

	if sh == nil {
		return 0
	}

	// Followed by OpenBSD string and version.
	Cwrite(ELF_NOTE_OPENBSD_NAME)

	Thearch.Lput(ELF_NOTE_OPENBSD_VERSION)

	return int(sh.size)
}

func addbuildinfo(val string) {
	if !strings.HasPrefix(val, "0x") {
		Exitf("-B argument must start with 0x: %s", val)
	}

	ov := val
	val = val[2:]

	const maxLen = 32
	if hex.DecodedLen(len(val)) > maxLen {
		Exitf("-B option too long (max %d digits): %s", maxLen, ov)
	}

	b, err := hex.DecodeString(val)
	if err != nil {
		if err == hex.ErrLength {
			Exitf("-B argument must have even number of digits: %s", ov)
		}
		if inv, ok := err.(hex.InvalidByteError); ok {
			Exitf("-B argument contains invalid hex digit %c: %s", byte(inv), ov)
		}
		Exitf("-B argument contains invalid hex: %s", ov)
	}

	buildinfo = b
}

// Build info note
const (
	ELF_NOTE_BUILDINFO_NAMESZ = 4
	ELF_NOTE_BUILDINFO_TAG    = 3
)

var ELF_NOTE_BUILDINFO_NAME = []byte("GNU\x00")

func elfbuildinfo(sh *ElfShdr, startva uint64, resoff uint64) int {
	n := int(ELF_NOTE_BUILDINFO_NAMESZ + Rnd(int64(len(buildinfo)), 4))
	return elfnote(sh, startva, resoff, n, true)
}

func elfgobuildid(sh *ElfShdr, startva uint64, resoff uint64) int {
	n := len(ELF_NOTE_GO_NAME) + int(Rnd(int64(len(*flagBuildid)), 4))
	return elfnote(sh, startva, resoff, n, true)
}

func elfwritebuildinfo() int {
	sh := elfwritenotehdr(".note.gnu.build-id", ELF_NOTE_BUILDINFO_NAMESZ, uint32(len(buildinfo)), ELF_NOTE_BUILDINFO_TAG)
	if sh == nil {
		return 0
	}

	Cwrite(ELF_NOTE_BUILDINFO_NAME)
	Cwrite(buildinfo)
	var zero = make([]byte, 4)
	Cwrite(zero[:int(Rnd(int64(len(buildinfo)), 4)-int64(len(buildinfo)))])

	return int(sh.size)
}

func elfwritegobuildid() int {
	sh := elfwritenotehdr(".note.go.buildid", uint32(len(ELF_NOTE_GO_NAME)), uint32(len(*flagBuildid)), ELF_NOTE_GOBUILDID_TAG)
	if sh == nil {
		return 0
	}

	Cwrite(ELF_NOTE_GO_NAME)
	Cwrite([]byte(*flagBuildid))
	var zero = make([]byte, 4)
	Cwrite(zero[:int(Rnd(int64(len(*flagBuildid)), 4)-int64(len(*flagBuildid)))])

	return int(sh.size)
}

// Go specific notes
const (
	ELF_NOTE_GOPKGLIST_TAG = 1
	ELF_NOTE_GOABIHASH_TAG = 2
	ELF_NOTE_GODEPS_TAG    = 3
	ELF_NOTE_GOBUILDID_TAG = 4
)

var ELF_NOTE_GO_NAME = []byte("Go\x00\x00")

var elfverneed int

type Elfaux struct {
	next *Elfaux
	num  int
	vers string
}

type Elflib struct {
	next *Elflib
	aux  *Elfaux
	file string
}

func addelflib(list **Elflib, file string, vers string) *Elfaux {
	var lib *Elflib

	for lib = *list; lib != nil; lib = lib.next {
		if lib.file == file {
			goto havelib
		}
	}
	lib = new(Elflib)
	lib.next = *list
	lib.file = file
	*list = lib

havelib:
	for aux := lib.aux; aux != nil; aux = aux.next {
		if aux.vers == vers {
			return aux
		}
	}
	aux := new(Elfaux)
	aux.next = lib.aux
	aux.vers = vers
	lib.aux = aux

	return aux
}

func elfdynhash(ctxt *Link) {
	if !Iself {
		return
	}

	nsym := Nelfsym
	s := ctxt.Syms.Lookup(".hash", 0)
	s.Type = SELFROSECT
	s.Attr |= AttrReachable

	i := nsym
	nbucket := 1
	for i > 0 {
		nbucket++
		i >>= 1
	}

	var needlib *Elflib
	need := make([]*Elfaux, nsym)
	chain := make([]uint32, nsym)
	buckets := make([]uint32, nbucket)

	var b int
	for _, sy := range ctxt.Syms.Allsym {
		if sy.Dynid <= 0 {
			continue
		}

		if sy.Dynimpvers != "" {
			need[sy.Dynid] = addelflib(&needlib, sy.Dynimplib, sy.Dynimpvers)
		}

		name := sy.Extname
		hc := elfhash(name)

		b = int(hc % uint32(nbucket))
		chain[sy.Dynid] = buckets[b]
		buckets[b] = uint32(sy.Dynid)
	}

	// s390x (ELF64) hash table entries are 8 bytes
	if SysArch.Family == sys.S390X {
		Adduint64(ctxt, s, uint64(nbucket))
		Adduint64(ctxt, s, uint64(nsym))
		for i := 0; i < nbucket; i++ {
			Adduint64(ctxt, s, uint64(buckets[i]))
		}
		for i := 0; i < nsym; i++ {
			Adduint64(ctxt, s, uint64(chain[i]))
		}
	} else {
		Adduint32(ctxt, s, uint32(nbucket))
		Adduint32(ctxt, s, uint32(nsym))
		for i := 0; i < nbucket; i++ {
			Adduint32(ctxt, s, buckets[i])
		}
		for i := 0; i < nsym; i++ {
			Adduint32(ctxt, s, chain[i])
		}
	}

	// version symbols
	dynstr := ctxt.Syms.Lookup(".dynstr", 0)

	s = ctxt.Syms.Lookup(".gnu.version_r", 0)
	i = 2
	nfile := 0
	var j int
	var x *Elfaux
	for l := needlib; l != nil; l = l.next {
		nfile++

		// header
		Adduint16(ctxt, s, 1) // table version
		j = 0
		for x = l.aux; x != nil; x = x.next {
			j++
		}
		Adduint16(ctxt, s, uint16(j))                         // aux count
		Adduint32(ctxt, s, uint32(Addstring(dynstr, l.file))) // file string offset
		Adduint32(ctxt, s, 16)                                // offset from header to first aux
		if l.next != nil {
			Adduint32(ctxt, s, 16+uint32(j)*16) // offset from this header to next
		} else {
			Adduint32(ctxt, s, 0)
		}

		for x = l.aux; x != nil; x = x.next {
			x.num = i
			i++

			// aux struct
			Adduint32(ctxt, s, elfhash(x.vers))                   // hash
			Adduint16(ctxt, s, 0)                                 // flags
			Adduint16(ctxt, s, uint16(x.num))                     // other - index we refer to this by
			Adduint32(ctxt, s, uint32(Addstring(dynstr, x.vers))) // version string offset
			if x.next != nil {
				Adduint32(ctxt, s, 16) // offset from this aux to next
			} else {
				Adduint32(ctxt, s, 0)
			}
		}
	}

	// version references
	s = ctxt.Syms.Lookup(".gnu.version", 0)

	for i := 0; i < nsym; i++ {
		if i == 0 {
			Adduint16(ctxt, s, 0) // first entry - no symbol
		} else if need[i] == nil {
			Adduint16(ctxt, s, 1) // global
		} else {
			Adduint16(ctxt, s, uint16(need[i].num))
		}
	}

	s = ctxt.Syms.Lookup(".dynamic", 0)
	elfverneed = nfile
	if elfverneed != 0 {
		elfwritedynentsym(ctxt, s, DT_VERNEED, ctxt.Syms.Lookup(".gnu.version_r", 0))
		Elfwritedynent(ctxt, s, DT_VERNEEDNUM, uint64(nfile))
		elfwritedynentsym(ctxt, s, DT_VERSYM, ctxt.Syms.Lookup(".gnu.version", 0))
	}

	sy := ctxt.Syms.Lookup(elfRelType+".plt", 0)
	if sy.Size > 0 {
		if elfRelType == ".rela" {
			Elfwritedynent(ctxt, s, DT_PLTREL, DT_RELA)
		} else {
			Elfwritedynent(ctxt, s, DT_PLTREL, DT_REL)
		}
		elfwritedynentsymsize(ctxt, s, DT_PLTRELSZ, sy)
		elfwritedynentsym(ctxt, s, DT_JMPREL, sy)
	}

	Elfwritedynent(ctxt, s, DT_NULL, 0)
}

func elfphload(seg *Segment) *ElfPhdr {
	ph := newElfPhdr()
	ph.type_ = PT_LOAD
	if seg.Rwx&4 != 0 {
		ph.flags |= PF_R
	}
	if seg.Rwx&2 != 0 {
		ph.flags |= PF_W
	}
	if seg.Rwx&1 != 0 {
		ph.flags |= PF_X
	}
	ph.vaddr = seg.Vaddr
	ph.paddr = seg.Vaddr
	ph.memsz = seg.Length
	ph.off = seg.Fileoff
	ph.filesz = seg.Filelen
	ph.align = uint64(*FlagRound)

	return ph
}

func elfphrelro(seg *Segment) {
	ph := newElfPhdr()
	ph.type_ = PT_GNU_RELRO
	ph.vaddr = seg.Vaddr
	ph.paddr = seg.Vaddr
	ph.memsz = seg.Length
	ph.off = seg.Fileoff
	ph.filesz = seg.Filelen
	ph.align = uint64(*FlagRound)
}

func elfshname(name string) *ElfShdr {
	var off int
	var sh *ElfShdr

	for i := 0; i < nelfstr; i++ {
		if name == elfstr[i].s {
			off = elfstr[i].off
			for i = 0; i < int(ehdr.shnum); i++ {
				sh = shdr[i]
				if sh.name == uint32(off) {
					return sh
				}
			}

			sh = newElfShdr(int64(off))
			return sh
		}
	}

	Exitf("cannot find elf name %s", name)
	return nil
}

// Create an ElfShdr for the section with name.
// Create a duplicate if one already exists with that name
func elfshnamedup(name string) *ElfShdr {
	var off int
	var sh *ElfShdr

	for i := 0; i < nelfstr; i++ {
		if name == elfstr[i].s {
			off = elfstr[i].off
			sh = newElfShdr(int64(off))
			return sh
		}
	}

	Errorf(nil, "cannot find elf name %s", name)
	errorexit()
	return nil
}

func elfshalloc(sect *Section) *ElfShdr {
	sh := elfshname(sect.Name)
	sect.Elfsect = sh
	return sh
}

func elfshbits(sect *Section) *ElfShdr {
	var sh *ElfShdr

	if sect.Name == ".text" {
		if sect.Elfsect == nil {
			sect.Elfsect = elfshnamedup(sect.Name)
		}
		sh = sect.Elfsect
	} else {
		sh = elfshalloc(sect)
	}

	// If this section has already been set up as a note, we assume type_ and
	// flags are already correct, but the other fields still need filling in.
	if sh.type_ == SHT_NOTE {
		if Linkmode != LinkExternal {
			// TODO(mwhudson): the approach here will work OK when
			// linking internally for notes that we want to be included
			// in a loadable segment (e.g. the abihash note) but not for
			// notes that we do not want to be mapped (e.g. the package
			// list note). The real fix is probably to define new values
			// for Symbol.Type corresponding to mapped and unmapped notes
			// and handle them in dodata().
			Errorf(nil, "sh.type_ == SHT_NOTE in elfshbits when linking internally")
		}
		sh.addralign = uint64(sect.Align)
		sh.size = sect.Length
		sh.off = sect.Seg.Fileoff + sect.Vaddr - sect.Seg.Vaddr
		return sh
	}
	if sh.type_ > 0 {
		return sh
	}

	if sect.Vaddr < sect.Seg.Vaddr+sect.Seg.Filelen {
		sh.type_ = SHT_PROGBITS
	} else {
		sh.type_ = SHT_NOBITS
	}
	sh.flags = SHF_ALLOC
	if sect.Rwx&1 != 0 {
		sh.flags |= SHF_EXECINSTR
	}
	if sect.Rwx&2 != 0 {
		sh.flags |= SHF_WRITE
	}
	if sect.Name == ".tbss" {
		sh.flags |= SHF_TLS
		sh.type_ = SHT_NOBITS
	}
	if strings.HasPrefix(sect.Name, ".debug") {
		sh.flags = 0
	}

	if Linkmode != LinkExternal {
		sh.addr = sect.Vaddr
	}
	sh.addralign = uint64(sect.Align)
	sh.size = sect.Length
	if sect.Name != ".tbss" {
		sh.off = sect.Seg.Fileoff + sect.Vaddr - sect.Seg.Vaddr
	}

	return sh
}

func elfshreloc(sect *Section) *ElfShdr {
	// If main section is SHT_NOBITS, nothing to relocate.
	// Also nothing to relocate in .shstrtab or notes.
	if sect.Vaddr >= sect.Seg.Vaddr+sect.Seg.Filelen {
		return nil
	}
	if sect.Name == ".shstrtab" || sect.Name == ".tbss" {
		return nil
	}
	if sect.Elfsect.type_ == SHT_NOTE {
		return nil
	}

	var typ int
	if elfRelType == ".rela" {
		typ = SHT_RELA
	} else {
		typ = SHT_REL
	}

	sh := elfshname(elfRelType + sect.Name)
	// There could be multiple text sections but each needs
	// its own .rela.text.

	if sect.Name == ".text" {
		if sh.info != 0 && sh.info != uint32(sect.Elfsect.shnum) {
			sh = elfshnamedup(elfRelType + sect.Name)
		}
	}

	sh.type_ = uint32(typ)
	sh.entsize = uint64(SysArch.RegSize) * 2
	if typ == SHT_RELA {
		sh.entsize += uint64(SysArch.RegSize)
	}
	sh.link = uint32(elfshname(".symtab").shnum)
	sh.info = uint32(sect.Elfsect.shnum)
	sh.off = sect.Reloff
	sh.size = sect.Rellen
	sh.addralign = uint64(SysArch.RegSize)
	return sh
}

func elfrelocsect(ctxt *Link, sect *Section, syms []*Symbol) {
	// If main section is SHT_NOBITS, nothing to relocate.
	// Also nothing to relocate in .shstrtab.
	if sect.Vaddr >= sect.Seg.Vaddr+sect.Seg.Filelen {
		return
	}
	if sect.Name == ".shstrtab" {
		return
	}

	sect.Reloff = uint64(coutbuf.Offset())
	for i, s := range syms {
		if !s.Attr.Reachable() {
			continue
		}
		if uint64(s.Value) >= sect.Vaddr {
			syms = syms[i:]
			break
		}
	}

	eaddr := int32(sect.Vaddr + sect.Length)
	for _, sym := range syms {
		if !sym.Attr.Reachable() {
			continue
		}
		if sym.Value >= int64(eaddr) {
			break
		}
		for ri := 0; ri < len(sym.R); ri++ {
			r := &sym.R[ri]
			if r.Done != 0 {
				continue
			}
			if r.Xsym == nil {
				Errorf(sym, "missing xsym in relocation")
				continue
			}
			if r.Xsym.ElfsymForReloc() == 0 {
				Errorf(sym, "reloc %d to non-elf symbol %s (outer=%s) %d", r.Type, r.Sym.Name, r.Xsym.Name, r.Sym.Type)
			}
			if !r.Xsym.Attr.Reachable() {
				Errorf(sym, "unreachable reloc %v target %v", r.Type, r.Xsym.Name)
			}
			if Thearch.Elfreloc1(ctxt, r, int64(uint64(sym.Value+int64(r.Off))-sect.Vaddr)) < 0 {
				Errorf(sym, "unsupported obj reloc %d/%d to %s", r.Type, r.Siz, r.Sym.Name)
			}
		}
	}

	sect.Rellen = uint64(coutbuf.Offset()) - sect.Reloff
}

func Elfemitreloc(ctxt *Link) {
	for coutbuf.Offset()&7 != 0 {
		Cput(0)
	}

	for _, sect := range Segtext.Sections {
		if sect.Name == ".text" {
			elfrelocsect(ctxt, sect, ctxt.Textp)
		} else {
			elfrelocsect(ctxt, sect, datap)
		}
	}

	for _, sect := range Segrodata.Sections {
		elfrelocsect(ctxt, sect, datap)
	}
	for _, sect := range Segrelrodata.Sections {
		elfrelocsect(ctxt, sect, datap)
	}
	for _, sect := range Segdata.Sections {
		elfrelocsect(ctxt, sect, datap)
	}
	for _, sect := range Segdwarf.Sections {
		elfrelocsect(ctxt, sect, dwarfp)
	}
}

func addgonote(ctxt *Link, sectionName string, tag uint32, desc []byte) {
	s := ctxt.Syms.Lookup(sectionName, 0)
	s.Attr |= AttrReachable
	s.Type = SELFROSECT
	// namesz
	Adduint32(ctxt, s, uint32(len(ELF_NOTE_GO_NAME)))
	// descsz
	Adduint32(ctxt, s, uint32(len(desc)))
	// tag
	Adduint32(ctxt, s, tag)
	// name + padding
	s.P = append(s.P, ELF_NOTE_GO_NAME...)
	for len(s.P)%4 != 0 {
		s.P = append(s.P, 0)
	}
	// desc + padding
	s.P = append(s.P, desc...)
	for len(s.P)%4 != 0 {
		s.P = append(s.P, 0)
	}
	s.Size = int64(len(s.P))
}

func (ctxt *Link) doelf() {
	if !Iself {
		return
	}

	/* predefine strings we need for section headers */
	shstrtab := ctxt.Syms.Lookup(".shstrtab", 0)

	shstrtab.Type = SELFROSECT
	shstrtab.Attr |= AttrReachable

	Addstring(shstrtab, "")
	Addstring(shstrtab, ".text")
	Addstring(shstrtab, ".noptrdata")
	Addstring(shstrtab, ".data")
	Addstring(shstrtab, ".bss")
	Addstring(shstrtab, ".noptrbss")

	// generate .tbss section for dynamic internal linker or external
	// linking, so that various binutils could correctly calculate
	// PT_TLS size. See https://golang.org/issue/5200.
	if !*FlagD || Linkmode == LinkExternal {
		Addstring(shstrtab, ".tbss")
	}
	if Headtype == objabi.Hnetbsd {
		Addstring(shstrtab, ".note.netbsd.ident")
	}
	if Headtype == objabi.Hopenbsd {
		Addstring(shstrtab, ".note.openbsd.ident")
	}
	if len(buildinfo) > 0 {
		Addstring(shstrtab, ".note.gnu.build-id")
	}
	if *flagBuildid != "" {
		Addstring(shstrtab, ".note.go.buildid")
	}
	Addstring(shstrtab, ".elfdata")
	Addstring(shstrtab, ".rodata")
	// See the comment about data.rel.ro.FOO section names in data.go.
	relro_prefix := ""
	if UseRelro() {
		Addstring(shstrtab, ".data.rel.ro")
		relro_prefix = ".data.rel.ro"
	}
	Addstring(shstrtab, relro_prefix+".typelink")
	Addstring(shstrtab, relro_prefix+".itablink")
	Addstring(shstrtab, relro_prefix+".gosymtab")
	Addstring(shstrtab, relro_prefix+".gopclntab")

	if Linkmode == LinkExternal {
		*FlagD = true

		Addstring(shstrtab, elfRelType+".text")
		Addstring(shstrtab, elfRelType+".rodata")
		Addstring(shstrtab, elfRelType+relro_prefix+".typelink")
		Addstring(shstrtab, elfRelType+relro_prefix+".itablink")
		Addstring(shstrtab, elfRelType+relro_prefix+".gosymtab")
		Addstring(shstrtab, elfRelType+relro_prefix+".gopclntab")
		Addstring(shstrtab, elfRelType+".noptrdata")
		Addstring(shstrtab, elfRelType+".data")
		if UseRelro() {
			Addstring(shstrtab, elfRelType+".data.rel.ro")
		}

		// add a .note.GNU-stack section to mark the stack as non-executable
		Addstring(shstrtab, ".note.GNU-stack")

		if Buildmode == BuildmodeShared {
			Addstring(shstrtab, ".note.go.abihash")
			Addstring(shstrtab, ".note.go.pkg-list")
			Addstring(shstrtab, ".note.go.deps")
		}
	}

	hasinitarr := *FlagLinkshared

	/* shared library initializer */
	switch Buildmode {
	case BuildmodeCArchive, BuildmodeCShared, BuildmodeShared, BuildmodePlugin:
		hasinitarr = true
	}

	if hasinitarr {
		Addstring(shstrtab, ".init_array")
		Addstring(shstrtab, elfRelType+".init_array")
	}

	if !*FlagS {
		Addstring(shstrtab, ".symtab")
		Addstring(shstrtab, ".strtab")
		dwarfaddshstrings(ctxt, shstrtab)
	}

	Addstring(shstrtab, ".shstrtab")

	if !*FlagD { /* -d suppresses dynamic loader format */
		Addstring(shstrtab, ".interp")
		Addstring(shstrtab, ".hash")
		Addstring(shstrtab, ".got")
		if SysArch.Family == sys.PPC64 {
			Addstring(shstrtab, ".glink")
		}
		Addstring(shstrtab, ".got.plt")
		Addstring(shstrtab, ".dynamic")
		Addstring(shstrtab, ".dynsym")
		Addstring(shstrtab, ".dynstr")
		Addstring(shstrtab, elfRelType)
		Addstring(shstrtab, elfRelType+".plt")

		Addstring(shstrtab, ".plt")
		Addstring(shstrtab, ".gnu.version")
		Addstring(shstrtab, ".gnu.version_r")

		/* dynamic symbol table - first entry all zeros */
		s := ctxt.Syms.Lookup(".dynsym", 0)

		s.Type = SELFROSECT
		s.Attr |= AttrReachable
		if elf64 {
			s.Size += ELF64SYMSIZE
		} else {
			s.Size += ELF32SYMSIZE
		}

		/* dynamic string table */
		s = ctxt.Syms.Lookup(".dynstr", 0)

		s.Type = SELFROSECT
		s.Attr |= AttrReachable
		if s.Size == 0 {
			Addstring(s, "")
		}
		dynstr := s

		/* relocation table */
		s = ctxt.Syms.Lookup(elfRelType, 0)
		s.Attr |= AttrReachable
		s.Type = SELFROSECT

		/* global offset table */
		s = ctxt.Syms.Lookup(".got", 0)

		s.Attr |= AttrReachable
		s.Type = SELFGOT // writable

		/* ppc64 glink resolver */
		if SysArch.Family == sys.PPC64 {
			s := ctxt.Syms.Lookup(".glink", 0)
			s.Attr |= AttrReachable
			s.Type = SELFRXSECT
		}

		/* hash */
		s = ctxt.Syms.Lookup(".hash", 0)

		s.Attr |= AttrReachable
		s.Type = SELFROSECT

		s = ctxt.Syms.Lookup(".got.plt", 0)
		s.Attr |= AttrReachable
		s.Type = SELFSECT // writable

		s = ctxt.Syms.Lookup(".plt", 0)

		s.Attr |= AttrReachable
		if SysArch.Family == sys.PPC64 {
			// In the ppc64 ABI, .plt is a data section
			// written by the dynamic linker.
			s.Type = SELFSECT
		} else {
			s.Type = SELFRXSECT
		}

		Thearch.Elfsetupplt(ctxt)

		s = ctxt.Syms.Lookup(elfRelType+".plt", 0)
		s.Attr |= AttrReachable
		s.Type = SELFROSECT

		s = ctxt.Syms.Lookup(".gnu.version", 0)
		s.Attr |= AttrReachable
		s.Type = SELFROSECT

		s = ctxt.Syms.Lookup(".gnu.version_r", 0)
		s.Attr |= AttrReachable
		s.Type = SELFROSECT

		/* define dynamic elf table */
		s = ctxt.Syms.Lookup(".dynamic", 0)

		s.Attr |= AttrReachable
		s.Type = SELFSECT // writable

		/*
		 * .dynamic table
		 */
		elfwritedynentsym(ctxt, s, DT_HASH, ctxt.Syms.Lookup(".hash", 0))

		elfwritedynentsym(ctxt, s, DT_SYMTAB, ctxt.Syms.Lookup(".dynsym", 0))
		if elf64 {
			Elfwritedynent(ctxt, s, DT_SYMENT, ELF64SYMSIZE)
		} else {
			Elfwritedynent(ctxt, s, DT_SYMENT, ELF32SYMSIZE)
		}
		elfwritedynentsym(ctxt, s, DT_STRTAB, ctxt.Syms.Lookup(".dynstr", 0))
		elfwritedynentsymsize(ctxt, s, DT_STRSZ, ctxt.Syms.Lookup(".dynstr", 0))
		if elfRelType == ".rela" {
			elfwritedynentsym(ctxt, s, DT_RELA, ctxt.Syms.Lookup(".rela", 0))
			elfwritedynentsymsize(ctxt, s, DT_RELASZ, ctxt.Syms.Lookup(".rela", 0))
			Elfwritedynent(ctxt, s, DT_RELAENT, ELF64RELASIZE)
		} else {
			elfwritedynentsym(ctxt, s, DT_REL, ctxt.Syms.Lookup(".rel", 0))
			elfwritedynentsymsize(ctxt, s, DT_RELSZ, ctxt.Syms.Lookup(".rel", 0))
			Elfwritedynent(ctxt, s, DT_RELENT, ELF32RELSIZE)
		}

		if rpath.val != "" {
			Elfwritedynent(ctxt, s, DT_RUNPATH, uint64(Addstring(dynstr, rpath.val)))
		}

		if SysArch.Family == sys.PPC64 {
			elfwritedynentsym(ctxt, s, DT_PLTGOT, ctxt.Syms.Lookup(".plt", 0))
		} else if SysArch.Family == sys.S390X {
			elfwritedynentsym(ctxt, s, DT_PLTGOT, ctxt.Syms.Lookup(".got", 0))
		} else {
			elfwritedynentsym(ctxt, s, DT_PLTGOT, ctxt.Syms.Lookup(".got.plt", 0))
		}

		if SysArch.Family == sys.PPC64 {
			Elfwritedynent(ctxt, s, DT_PPC64_OPT, 0)
		}

		// Solaris dynamic linker can't handle an empty .rela.plt if
		// DT_JMPREL is emitted so we have to defer generation of DT_PLTREL,
		// DT_PLTRELSZ, and DT_JMPREL dynamic entries until after we know the
		// size of .rel(a).plt section.
		Elfwritedynent(ctxt, s, DT_DEBUG, 0)
	}

	if Buildmode == BuildmodeShared {
		// The go.link.abihashbytes symbol will be pointed at the appropriate
		// part of the .note.go.abihash section in data.go:func address().
		s := ctxt.Syms.Lookup("go.link.abihashbytes", 0)
		s.Attr |= AttrLocal
		s.Type = SRODATA
		s.Attr |= AttrSpecial
		s.Attr |= AttrReachable
		s.Size = int64(sha1.Size)

		sort.Sort(byPkg(ctxt.Library))
		h := sha1.New()
		for _, l := range ctxt.Library {
			io.WriteString(h, l.hash)
		}
		addgonote(ctxt, ".note.go.abihash", ELF_NOTE_GOABIHASH_TAG, h.Sum([]byte{}))
		addgonote(ctxt, ".note.go.pkg-list", ELF_NOTE_GOPKGLIST_TAG, pkglistfornote)
		var deplist []string
		for _, shlib := range ctxt.Shlibs {
			deplist = append(deplist, filepath.Base(shlib.Path))
		}
		addgonote(ctxt, ".note.go.deps", ELF_NOTE_GODEPS_TAG, []byte(strings.Join(deplist, "\n")))
	}

	if Linkmode == LinkExternal && *flagBuildid != "" {
		addgonote(ctxt, ".note.go.buildid", ELF_NOTE_GOBUILDID_TAG, []byte(*flagBuildid))
	}
}

// Do not write DT_NULL.  elfdynhash will finish it.
func shsym(sh *ElfShdr, s *Symbol) {
	addr := Symaddr(s)
	if sh.flags&SHF_ALLOC != 0 {
		sh.addr = uint64(addr)
	}
	sh.off = uint64(datoff(s, addr))
	sh.size = uint64(s.Size)
}

func phsh(ph *ElfPhdr, sh *ElfShdr) {
	ph.vaddr = sh.addr
	ph.paddr = ph.vaddr
	ph.off = sh.off
	ph.filesz = sh.size
	ph.memsz = sh.size
	ph.align = sh.addralign
}

func Asmbelfsetup() {
	/* This null SHdr must appear before all others */
	elfshname("")

	for _, sect := range Segtext.Sections {
		// There could be multiple .text sections. Instead check the Elfsect
		// field to determine if already has an ElfShdr and if not, create one.
		if sect.Name == ".text" {
			if sect.Elfsect == nil {
				sect.Elfsect = elfshnamedup(sect.Name)
			}
		} else {
			elfshalloc(sect)
		}
	}
	for _, sect := range Segrodata.Sections {
		elfshalloc(sect)
	}
	for _, sect := range Segrelrodata.Sections {
		elfshalloc(sect)
	}
	for _, sect := range Segdata.Sections {
		elfshalloc(sect)
	}
	for _, sect := range Segdwarf.Sections {
		elfshalloc(sect)
	}
}

func Asmbelf(ctxt *Link, symo int64) {
	eh := getElfEhdr()
	switch SysArch.Family {
	default:
		Exitf("unknown architecture in asmbelf: %v", SysArch.Family)
	case sys.MIPS, sys.MIPS64:
		eh.machine = EM_MIPS
	case sys.ARM:
		eh.machine = EM_ARM
	case sys.AMD64:
		eh.machine = EM_X86_64
	case sys.ARM64:
		eh.machine = EM_AARCH64
	case sys.I386:
		eh.machine = EM_386
	case sys.PPC64:
		eh.machine = EM_PPC64
	case sys.S390X:
		eh.machine = EM_S390
	}

	elfreserve := int64(ELFRESERVE)

	numtext := int64(0)
	for _, sect := range Segtext.Sections {
		if sect.Name == ".text" {
			numtext++
		}
	}

	// If there are multiple text sections, extra space is needed
	// in the elfreserve for the additional .text and .rela.text
	// section headers.  It can handle 4 extra now. Headers are
	// 64 bytes.

	if numtext > 4 {
		elfreserve += elfreserve + numtext*64*2
	}

	startva := *FlagTextAddr - int64(HEADR)
	resoff := elfreserve

	var pph *ElfPhdr
	var pnote *ElfPhdr
	if Linkmode == LinkExternal {
		/* skip program headers */
		eh.phoff = 0

		eh.phentsize = 0

		if Buildmode == BuildmodeShared {
			sh := elfshname(".note.go.pkg-list")
			sh.type_ = SHT_NOTE
			sh = elfshname(".note.go.abihash")
			sh.type_ = SHT_NOTE
			sh.flags = SHF_ALLOC
			sh = elfshname(".note.go.deps")
			sh.type_ = SHT_NOTE
		}

		if *flagBuildid != "" {
			sh := elfshname(".note.go.buildid")
			sh.type_ = SHT_NOTE
			sh.flags = SHF_ALLOC
		}

		goto elfobj
	}

	/* program header info */
	pph = newElfPhdr()

	pph.type_ = PT_PHDR
	pph.flags = PF_R
	pph.off = uint64(eh.ehsize)
	pph.vaddr = uint64(*FlagTextAddr) - uint64(HEADR) + pph.off
	pph.paddr = uint64(*FlagTextAddr) - uint64(HEADR) + pph.off
	pph.align = uint64(*FlagRound)

	/*
	 * PHDR must be in a loaded segment. Adjust the text
	 * segment boundaries downwards to include it.
	 * Except on NaCl where it must not be loaded.
	 */
	if Headtype != objabi.Hnacl {
		o := int64(Segtext.Vaddr - pph.vaddr)
		Segtext.Vaddr -= uint64(o)
		Segtext.Length += uint64(o)
		o = int64(Segtext.Fileoff - pph.off)
		Segtext.Fileoff -= uint64(o)
		Segtext.Filelen += uint64(o)
	}

	if !*FlagD { /* -d suppresses dynamic loader format */
		/* interpreter */
		sh := elfshname(".interp")

		sh.type_ = SHT_PROGBITS
		sh.flags = SHF_ALLOC
		sh.addralign = 1
		if interpreter == "" {
			switch Headtype {
			case objabi.Hlinux:
				interpreter = Thearch.Linuxdynld

			case objabi.Hfreebsd:
				interpreter = Thearch.Freebsddynld

			case objabi.Hnetbsd:
				interpreter = Thearch.Netbsddynld

			case objabi.Hopenbsd:
				interpreter = Thearch.Openbsddynld

			case objabi.Hdragonfly:
				interpreter = Thearch.Dragonflydynld

			case objabi.Hsolaris:
				interpreter = Thearch.Solarisdynld
			}
		}

		resoff -= int64(elfinterp(sh, uint64(startva), uint64(resoff), interpreter))

		ph := newElfPhdr()
		ph.type_ = PT_INTERP
		ph.flags = PF_R
		phsh(ph, sh)
	}

	pnote = nil
	if Headtype == objabi.Hnetbsd || Headtype == objabi.Hopenbsd {
		var sh *ElfShdr
		switch Headtype {
		case objabi.Hnetbsd:
			sh = elfshname(".note.netbsd.ident")
			resoff -= int64(elfnetbsdsig(sh, uint64(startva), uint64(resoff)))

		case objabi.Hopenbsd:
			sh = elfshname(".note.openbsd.ident")
			resoff -= int64(elfopenbsdsig(sh, uint64(startva), uint64(resoff)))
		}

		pnote = newElfPhdr()
		pnote.type_ = PT_NOTE
		pnote.flags = PF_R
		phsh(pnote, sh)
	}

	if len(buildinfo) > 0 {
		sh := elfshname(".note.gnu.build-id")
		resoff -= int64(elfbuildinfo(sh, uint64(startva), uint64(resoff)))

		if pnote == nil {
			pnote = newElfPhdr()
			pnote.type_ = PT_NOTE
			pnote.flags = PF_R
		}

		phsh(pnote, sh)
	}

	if *flagBuildid != "" {
		sh := elfshname(".note.go.buildid")
		resoff -= int64(elfgobuildid(sh, uint64(startva), uint64(resoff)))

		pnote := newElfPhdr()
		pnote.type_ = PT_NOTE
		pnote.flags = PF_R
		phsh(pnote, sh)
	}

	// Additions to the reserved area must be above this line.

	elfphload(&Segtext)
	if len(Segrodata.Sections) > 0 {
		elfphload(&Segrodata)
	}
	if len(Segrelrodata.Sections) > 0 {
		elfphload(&Segrelrodata)
		elfphrelro(&Segrelrodata)
	}
	elfphload(&Segdata)

	/* Dynamic linking sections */
	if !*FlagD {
		sh := elfshname(".dynsym")
		sh.type_ = SHT_DYNSYM
		sh.flags = SHF_ALLOC
		if elf64 {
			sh.entsize = ELF64SYMSIZE
		} else {
			sh.entsize = ELF32SYMSIZE
		}
		sh.addralign = uint64(SysArch.RegSize)
		sh.link = uint32(elfshname(".dynstr").shnum)

		// sh->info = index of first non-local symbol (number of local symbols)
		shsym(sh, ctxt.Syms.Lookup(".dynsym", 0))

		sh = elfshname(".dynstr")
		sh.type_ = SHT_STRTAB
		sh.flags = SHF_ALLOC
		sh.addralign = 1
		shsym(sh, ctxt.Syms.Lookup(".dynstr", 0))

		if elfverneed != 0 {
			sh := elfshname(".gnu.version")
			sh.type_ = SHT_GNU_VERSYM
			sh.flags = SHF_ALLOC
			sh.addralign = 2
			sh.link = uint32(elfshname(".dynsym").shnum)
			sh.entsize = 2
			shsym(sh, ctxt.Syms.Lookup(".gnu.version", 0))

			sh = elfshname(".gnu.version_r")
			sh.type_ = SHT_GNU_VERNEED
			sh.flags = SHF_ALLOC
			sh.addralign = uint64(SysArch.RegSize)
			sh.info = uint32(elfverneed)
			sh.link = uint32(elfshname(".dynstr").shnum)
			shsym(sh, ctxt.Syms.Lookup(".gnu.version_r", 0))
		}

		if elfRelType == ".rela" {
			sh := elfshname(".rela.plt")
			sh.type_ = SHT_RELA
			sh.flags = SHF_ALLOC
			sh.entsize = ELF64RELASIZE
			sh.addralign = uint64(SysArch.RegSize)
			sh.link = uint32(elfshname(".dynsym").shnum)
			sh.info = uint32(elfshname(".plt").shnum)
			shsym(sh, ctxt.Syms.Lookup(".rela.plt", 0))

			sh = elfshname(".rela")
			sh.type_ = SHT_RELA
			sh.flags = SHF_ALLOC
			sh.entsize = ELF64RELASIZE
			sh.addralign = 8
			sh.link = uint32(elfshname(".dynsym").shnum)
			shsym(sh, ctxt.Syms.Lookup(".rela", 0))
		} else {
			sh := elfshname(".rel.plt")
			sh.type_ = SHT_REL
			sh.flags = SHF_ALLOC
			sh.entsize = ELF32RELSIZE
			sh.addralign = 4
			sh.link = uint32(elfshname(".dynsym").shnum)
			shsym(sh, ctxt.Syms.Lookup(".rel.plt", 0))

			sh = elfshname(".rel")
			sh.type_ = SHT_REL
			sh.flags = SHF_ALLOC
			sh.entsize = ELF32RELSIZE
			sh.addralign = 4
			sh.link = uint32(elfshname(".dynsym").shnum)
			shsym(sh, ctxt.Syms.Lookup(".rel", 0))
		}

		if eh.machine == EM_PPC64 {
			sh := elfshname(".glink")
			sh.type_ = SHT_PROGBITS
			sh.flags = SHF_ALLOC + SHF_EXECINSTR
			sh.addralign = 4
			shsym(sh, ctxt.Syms.Lookup(".glink", 0))
		}

		sh = elfshname(".plt")
		sh.type_ = SHT_PROGBITS
		sh.flags = SHF_ALLOC + SHF_EXECINSTR
		if eh.machine == EM_X86_64 {
			sh.entsize = 16
		} else if eh.machine == EM_S390 {
			sh.entsize = 32
		} else if eh.machine == EM_PPC64 {
			// On ppc64, this is just a table of addresses
			// filled by the dynamic linker
			sh.type_ = SHT_NOBITS

			sh.flags = SHF_ALLOC + SHF_WRITE
			sh.entsize = 8
		} else {
			sh.entsize = 4
		}
		sh.addralign = sh.entsize
		shsym(sh, ctxt.Syms.Lookup(".plt", 0))

		// On ppc64, .got comes from the input files, so don't
		// create it here, and .got.plt is not used.
		if eh.machine != EM_PPC64 {
			sh := elfshname(".got")
			sh.type_ = SHT_PROGBITS
			sh.flags = SHF_ALLOC + SHF_WRITE
			sh.entsize = uint64(SysArch.RegSize)
			sh.addralign = uint64(SysArch.RegSize)
			shsym(sh, ctxt.Syms.Lookup(".got", 0))

			sh = elfshname(".got.plt")
			sh.type_ = SHT_PROGBITS
			sh.flags = SHF_ALLOC + SHF_WRITE
			sh.entsize = uint64(SysArch.RegSize)
			sh.addralign = uint64(SysArch.RegSize)
			shsym(sh, ctxt.Syms.Lookup(".got.plt", 0))
		}

		sh = elfshname(".hash")
		sh.type_ = SHT_HASH
		sh.flags = SHF_ALLOC
		sh.entsize = 4
		sh.addralign = uint64(SysArch.RegSize)
		sh.link = uint32(elfshname(".dynsym").shnum)
		shsym(sh, ctxt.Syms.Lookup(".hash", 0))

		/* sh and PT_DYNAMIC for .dynamic section */
		sh = elfshname(".dynamic")

		sh.type_ = SHT_DYNAMIC
		sh.flags = SHF_ALLOC + SHF_WRITE
		sh.entsize = 2 * uint64(SysArch.RegSize)
		sh.addralign = uint64(SysArch.RegSize)
		sh.link = uint32(elfshname(".dynstr").shnum)
		shsym(sh, ctxt.Syms.Lookup(".dynamic", 0))
		ph := newElfPhdr()
		ph.type_ = PT_DYNAMIC
		ph.flags = PF_R + PF_W
		phsh(ph, sh)

		/*
		 * Thread-local storage segment (really just size).
		 */
		tlssize := uint64(0)
		for _, sect := range Segdata.Sections {
			if sect.Name == ".tbss" {
				tlssize = sect.Length
			}
		}
		if tlssize != 0 {
			ph := newElfPhdr()
			ph.type_ = PT_TLS
			ph.flags = PF_R
			ph.memsz = tlssize
			ph.align = uint64(SysArch.RegSize)
		}
	}

	if Headtype == objabi.Hlinux {
		ph := newElfPhdr()
		ph.type_ = PT_GNU_STACK
		ph.flags = PF_W + PF_R
		ph.align = uint64(SysArch.RegSize)

		ph = newElfPhdr()
		ph.type_ = PT_PAX_FLAGS
		ph.flags = 0x2a00 // mprotect, randexec, emutramp disabled
		ph.align = uint64(SysArch.RegSize)
	} else if Headtype == objabi.Hsolaris {
		ph := newElfPhdr()
		ph.type_ = PT_SUNWSTACK
		ph.flags = PF_W + PF_R
	}

elfobj:
	sh := elfshname(".shstrtab")
	sh.type_ = SHT_STRTAB
	sh.addralign = 1
	shsym(sh, ctxt.Syms.Lookup(".shstrtab", 0))
	eh.shstrndx = uint16(sh.shnum)

	// put these sections early in the list
	if !*FlagS {
		elfshname(".symtab")
		elfshname(".strtab")
	}

	for _, sect := range Segtext.Sections {
		elfshbits(sect)
	}
	for _, sect := range Segrodata.Sections {
		elfshbits(sect)
	}
	for _, sect := range Segrelrodata.Sections {
		elfshbits(sect)
	}
	for _, sect := range Segdata.Sections {
		elfshbits(sect)
	}
	for _, sect := range Segdwarf.Sections {
		elfshbits(sect)
	}

	if Linkmode == LinkExternal {
		for _, sect := range Segtext.Sections {
			elfshreloc(sect)
		}
		for _, sect := range Segrodata.Sections {
			elfshreloc(sect)
		}
		for _, sect := range Segrelrodata.Sections {
			elfshreloc(sect)
		}
		for _, sect := range Segdata.Sections {
			elfshreloc(sect)
		}
		for _, s := range dwarfp {
			if len(s.R) > 0 || s.Type == SDWARFINFO {
				elfshreloc(s.Sect)
			}
			if s.Type == SDWARFINFO {
				break
			}
		}
		// add a .note.GNU-stack section to mark the stack as non-executable
		sh := elfshname(".note.GNU-stack")

		sh.type_ = SHT_PROGBITS
		sh.addralign = 1
		sh.flags = 0
	}

	if !*FlagS {
		sh := elfshname(".symtab")
		sh.type_ = SHT_SYMTAB
		sh.off = uint64(symo)
		sh.size = uint64(Symsize)
		sh.addralign = uint64(SysArch.RegSize)
		sh.entsize = 8 + 2*uint64(SysArch.RegSize)
		sh.link = uint32(elfshname(".strtab").shnum)
		sh.info = uint32(elfglobalsymndx)

		sh = elfshname(".strtab")
		sh.type_ = SHT_STRTAB
		sh.off = uint64(symo) + uint64(Symsize)
		sh.size = uint64(len(Elfstrdat))
		sh.addralign = 1
	}

	/* Main header */
	eh.ident[EI_MAG0] = '\177'

	eh.ident[EI_MAG1] = 'E'
	eh.ident[EI_MAG2] = 'L'
	eh.ident[EI_MAG3] = 'F'
	if Headtype == objabi.Hfreebsd {
		eh.ident[EI_OSABI] = ELFOSABI_FREEBSD
	} else if Headtype == objabi.Hnetbsd {
		eh.ident[EI_OSABI] = ELFOSABI_NETBSD
	} else if Headtype == objabi.Hopenbsd {
		eh.ident[EI_OSABI] = ELFOSABI_OPENBSD
	} else if Headtype == objabi.Hdragonfly {
		eh.ident[EI_OSABI] = ELFOSABI_NONE
	}
	if elf64 {
		eh.ident[EI_CLASS] = ELFCLASS64
	} else {
		eh.ident[EI_CLASS] = ELFCLASS32
	}
	if ctxt.Arch.ByteOrder == binary.BigEndian {
		eh.ident[EI_DATA] = ELFDATA2MSB
	} else {
		eh.ident[EI_DATA] = ELFDATA2LSB
	}
	eh.ident[EI_VERSION] = EV_CURRENT

	if Linkmode == LinkExternal {
		eh.type_ = ET_REL
	} else if Buildmode == BuildmodePIE {
		eh.type_ = ET_DYN
	} else {
		eh.type_ = ET_EXEC
	}

	if Linkmode != LinkExternal {
		eh.entry = uint64(Entryvalue(ctxt))
	}

	eh.version = EV_CURRENT

	if pph != nil {
		pph.filesz = uint64(eh.phnum) * uint64(eh.phentsize)
		pph.memsz = pph.filesz
	}

	Cseek(0)
	a := int64(0)
	a += int64(elfwritehdr())
	a += int64(elfwritephdrs())
	a += int64(elfwriteshdrs())
	if !*FlagD {
		a += int64(elfwriteinterp())
	}
	if Linkmode != LinkExternal {
		if Headtype == objabi.Hnetbsd {
			a += int64(elfwritenetbsdsig())
		}
		if Headtype == objabi.Hopenbsd {
			a += int64(elfwriteopenbsdsig())
		}
		if len(buildinfo) > 0 {
			a += int64(elfwritebuildinfo())
		}
		if *flagBuildid != "" {
			a += int64(elfwritegobuildid())
		}
	}

	if a > elfreserve {
		Errorf(nil, "ELFRESERVE too small: %d > %d with %d text sections", a, elfreserve, numtext)
	}
}

func Elfadddynsym(ctxt *Link, s *Symbol) {
	if elf64 {
		s.Dynid = int32(Nelfsym)
		Nelfsym++

		d := ctxt.Syms.Lookup(".dynsym", 0)

		name := s.Extname
		Adduint32(ctxt, d, uint32(Addstring(ctxt.Syms.Lookup(".dynstr", 0), name)))

		/* type */
		t := STB_GLOBAL << 4

		if s.Attr.CgoExport() && s.Type&SMASK == STEXT {
			t |= STT_FUNC
		} else {
			t |= STT_OBJECT
		}
		Adduint8(ctxt, d, uint8(t))

		/* reserved */
		Adduint8(ctxt, d, 0)

		/* section where symbol is defined */
		if s.Type == SDYNIMPORT {
			Adduint16(ctxt, d, SHN_UNDEF)
		} else {
			Adduint16(ctxt, d, 1)
		}

		/* value */
		if s.Type == SDYNIMPORT {
			Adduint64(ctxt, d, 0)
		} else {
			Addaddr(ctxt, d, s)
		}

		/* size of object */
		Adduint64(ctxt, d, uint64(s.Size))

		if SysArch.Family == sys.AMD64 && !s.Attr.CgoExportDynamic() && s.Dynimplib != "" && !seenlib[s.Dynimplib] {
			Elfwritedynent(ctxt, ctxt.Syms.Lookup(".dynamic", 0), DT_NEEDED, uint64(Addstring(ctxt.Syms.Lookup(".dynstr", 0), s.Dynimplib)))
		}
	} else {
		s.Dynid = int32(Nelfsym)
		Nelfsym++

		d := ctxt.Syms.Lookup(".dynsym", 0)

		/* name */
		name := s.Extname

		Adduint32(ctxt, d, uint32(Addstring(ctxt.Syms.Lookup(".dynstr", 0), name)))

		/* value */
		if s.Type == SDYNIMPORT {
			Adduint32(ctxt, d, 0)
		} else {
			Addaddr(ctxt, d, s)
		}

		/* size of object */
		Adduint32(ctxt, d, uint32(s.Size))

		/* type */
		t := STB_GLOBAL << 4

		// TODO(mwhudson): presumably the behavior should actually be the same on both arm and 386.
		if SysArch.Family == sys.I386 && s.Attr.CgoExport() && s.Type&SMASK == STEXT {
			t |= STT_FUNC
		} else if SysArch.Family == sys.ARM && s.Attr.CgoExportDynamic() && s.Type&SMASK == STEXT {
			t |= STT_FUNC
		} else {
			t |= STT_OBJECT
		}
		Adduint8(ctxt, d, uint8(t))
		Adduint8(ctxt, d, 0)

		/* shndx */
		if s.Type == SDYNIMPORT {
			Adduint16(ctxt, d, SHN_UNDEF)
		} else {
			Adduint16(ctxt, d, 1)
		}
	}
}

func ELF32_R_SYM(info uint32) uint32 {
	return info >> 8
}

func ELF32_R_TYPE(info uint32) uint32 {
	return uint32(uint8(info))
}

func ELF32_R_INFO(sym uint32, type_ uint32) uint32 {
	return sym<<8 | type_
}

func ELF32_ST_BIND(info uint8) uint8 {
	return info >> 4
}

func ELF32_ST_TYPE(info uint8) uint8 {
	return info & 0xf
}

func ELF32_ST_INFO(bind uint8, type_ uint8) uint8 {
	return bind<<4 | type_&0xf
}

func ELF32_ST_VISIBILITY(oth uint8) uint8 {
	return oth & 3
}

func ELF64_R_SYM(info uint64) uint32 {
	return uint32(info >> 32)
}

func ELF64_R_TYPE(info uint64) uint32 {
	return uint32(info)
}

func ELF64_R_INFO(sym uint32, type_ uint32) uint64 {
	return uint64(sym)<<32 | uint64(type_)
}

func ELF64_ST_BIND(info uint8) uint8 {
	return info >> 4
}

func ELF64_ST_TYPE(info uint8) uint8 {
	return info & 0xf
}

func ELF64_ST_INFO(bind uint8, type_ uint8) uint8 {
	return bind<<4 | type_&0xf
}

func ELF64_ST_VISIBILITY(oth uint8) uint8 {
	return oth & 3
}
