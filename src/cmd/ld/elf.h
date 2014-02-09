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

typedef struct {
	uint32	n_namesz;	/* Length of name. */
	uint32	n_descsz;	/* Length of descriptor. */
	uint32	n_type;		/* Type of this note. */
} Elf_Note;

/* Indexes into the e_ident array.  Keep synced with
   http://www.sco.com/developer/gabi/ch4.eheader.html */
#define EI_MAG0		0	/* Magic number, byte 0. */
#define EI_MAG1		1	/* Magic number, byte 1. */
#define EI_MAG2		2	/* Magic number, byte 2. */
#define EI_MAG3		3	/* Magic number, byte 3. */
#define EI_CLASS	4	/* Class of machine. */
#define EI_DATA		5	/* Data format. */
#define EI_VERSION	6	/* ELF format version. */
#define EI_OSABI	7	/* Operating system / ABI identification */
#define EI_ABIVERSION	8	/* ABI version */
#define OLD_EI_BRAND	8	/* Start of architecture identification. */
#define EI_PAD		9	/* Start of padding (per SVR4 ABI). */
#define EI_NIDENT	16	/* Size of e_ident array. */

/* Values for the magic number bytes. */
#define ELFMAG0		0x7f
#define ELFMAG1		'E'
#define ELFMAG2		'L'
#define ELFMAG3		'F'
#define ELFMAG		"\177ELF"	/* magic string */
#define SELFMAG		4		/* magic string size */

/* Values for e_ident[EI_VERSION] and e_version. */
#define EV_NONE		0
#define EV_CURRENT	1

/* Values for e_ident[EI_CLASS]. */
#define ELFCLASSNONE	0	/* Unknown class. */
#define ELFCLASS32	1	/* 32-bit architecture. */
#define ELFCLASS64	2	/* 64-bit architecture. */

/* Values for e_ident[EI_DATA]. */
#define ELFDATANONE	0	/* Unknown data format. */
#define ELFDATA2LSB	1	/* 2's complement little-endian. */
#define ELFDATA2MSB	2	/* 2's complement big-endian. */

/* Values for e_ident[EI_OSABI]. */
#define ELFOSABI_NONE		0	/* UNIX System V ABI */
#define ELFOSABI_HPUX		1	/* HP-UX operating system */
#define ELFOSABI_NETBSD		2	/* NetBSD */
#define ELFOSABI_LINUX		3	/* GNU/Linux */
#define ELFOSABI_HURD		4	/* GNU/Hurd */
#define ELFOSABI_86OPEN		5	/* 86Open common IA32 ABI */
#define ELFOSABI_SOLARIS	6	/* Solaris */
#define ELFOSABI_AIX		7	/* AIX */
#define ELFOSABI_IRIX		8	/* IRIX */
#define ELFOSABI_FREEBSD	9	/* FreeBSD */
#define ELFOSABI_TRU64		10	/* TRU64 UNIX */
#define ELFOSABI_MODESTO	11	/* Novell Modesto */
#define ELFOSABI_OPENBSD	12	/* OpenBSD */
#define ELFOSABI_OPENVMS	13	/* Open VMS */
#define ELFOSABI_NSK		14	/* HP Non-Stop Kernel */
#define ELFOSABI_ARM		97	/* ARM */
#define ELFOSABI_STANDALONE	255	/* Standalone (embedded) application */

#define ELFOSABI_SYSV		ELFOSABI_NONE	/* symbol used in old spec */
#define ELFOSABI_MONTEREY	ELFOSABI_AIX	/* Monterey */

/* e_ident */
#define IS_ELF(ehdr)	((ehdr).e_ident[EI_MAG0] == ELFMAG0 && \
			 (ehdr).e_ident[EI_MAG1] == ELFMAG1 && \
			 (ehdr).e_ident[EI_MAG2] == ELFMAG2 && \
			 (ehdr).e_ident[EI_MAG3] == ELFMAG3)

/* Values for e_type. */
#define ET_NONE		0	/* Unknown type. */
#define ET_REL		1	/* Relocatable. */
#define ET_EXEC		2	/* Executable. */
#define ET_DYN		3	/* Shared object. */
#define ET_CORE		4	/* Core file. */
#define ET_LOOS		0xfe00	/* First operating system specific. */
#define ET_HIOS		0xfeff	/* Last operating system-specific. */
#define ET_LOPROC	0xff00	/* First processor-specific. */
#define ET_HIPROC	0xffff	/* Last processor-specific. */

/* Values for e_machine. */
#define EM_NONE		0	/* Unknown machine. */
#define EM_M32		1	/* AT&T WE32100. */
#define EM_SPARC	2	/* Sun SPARC. */
#define EM_386		3	/* Intel i386. */
#define EM_68K		4	/* Motorola 68000. */
#define EM_88K		5	/* Motorola 88000. */
#define EM_860		7	/* Intel i860. */
#define EM_MIPS		8	/* MIPS R3000 Big-Endian only. */
#define EM_S370		9	/* IBM System/370. */
#define EM_MIPS_RS3_LE	10	/* MIPS R3000 Little-Endian. */
#define EM_PARISC	15	/* HP PA-RISC. */
#define EM_VPP500	17	/* Fujitsu VPP500. */
#define EM_SPARC32PLUS	18	/* SPARC v8plus. */
#define EM_960		19	/* Intel 80960. */
#define EM_PPC		20	/* PowerPC 32-bit. */
#define EM_PPC64	21	/* PowerPC 64-bit. */
#define EM_S390		22	/* IBM System/390. */
#define EM_V800		36	/* NEC V800. */
#define EM_FR20		37	/* Fujitsu FR20. */
#define EM_RH32		38	/* TRW RH-32. */
#define EM_RCE		39	/* Motorola RCE. */
#define EM_ARM		40	/* ARM. */
#define EM_SH		42	/* Hitachi SH. */
#define EM_SPARCV9	43	/* SPARC v9 64-bit. */
#define EM_TRICORE	44	/* Siemens TriCore embedded processor. */
#define EM_ARC		45	/* Argonaut RISC Core. */
#define EM_H8_300	46	/* Hitachi H8/300. */
#define EM_H8_300H	47	/* Hitachi H8/300H. */
#define EM_H8S		48	/* Hitachi H8S. */
#define EM_H8_500	49	/* Hitachi H8/500. */
#define EM_IA_64	50	/* Intel IA-64 Processor. */
#define EM_MIPS_X	51	/* Stanford MIPS-X. */
#define EM_COLDFIRE	52	/* Motorola ColdFire. */
#define EM_68HC12	53	/* Motorola M68HC12. */
#define EM_MMA		54	/* Fujitsu MMA. */
#define EM_PCP		55	/* Siemens PCP. */
#define EM_NCPU		56	/* Sony nCPU. */
#define EM_NDR1		57	/* Denso NDR1 microprocessor. */
#define EM_STARCORE	58	/* Motorola Star*Core processor. */
#define EM_ME16		59	/* Toyota ME16 processor. */
#define EM_ST100	60	/* STMicroelectronics ST100 processor. */
#define EM_TINYJ	61	/* Advanced Logic Corp. TinyJ processor. */
#define EM_X86_64	62	/* Advanced Micro Devices x86-64 */

/* Non-standard or deprecated. */
#define EM_486		6	/* Intel i486. */
#define EM_MIPS_RS4_BE	10	/* MIPS R4000 Big-Endian */
#define EM_ALPHA_STD	41	/* Digital Alpha (standard value). */
#define EM_ALPHA	0x9026	/* Alpha (written in the absence of an ABI) */

/* Special section indexes. */
#define SHN_UNDEF	     0		/* Undefined, missing, irrelevant. */
#define SHN_LORESERVE	0xff00		/* First of reserved range. */
#define SHN_LOPROC	0xff00		/* First processor-specific. */
#define SHN_HIPROC	0xff1f		/* Last processor-specific. */
#define SHN_LOOS	0xff20		/* First operating system-specific. */
#define SHN_HIOS	0xff3f		/* Last operating system-specific. */
#define SHN_ABS		0xfff1		/* Absolute values. */
#define SHN_COMMON	0xfff2		/* Common data. */
#define SHN_XINDEX	0xffff		/* Escape -- index stored elsewhere. */
#define SHN_HIRESERVE	0xffff		/* Last of reserved range. */

/* sh_type */
#define SHT_NULL		0	/* inactive */
#define SHT_PROGBITS		1	/* program defined information */
#define SHT_SYMTAB		2	/* symbol table section */
#define SHT_STRTAB		3	/* string table section */
#define SHT_RELA		4	/* relocation section with addends */
#define SHT_HASH		5	/* symbol hash table section */
#define SHT_DYNAMIC		6	/* dynamic section */
#define SHT_NOTE		7	/* note section */
#define SHT_NOBITS		8	/* no space section */
#define SHT_REL			9	/* relocation section - no addends */
#define SHT_SHLIB		10	/* reserved - purpose unknown */
#define SHT_DYNSYM		11	/* dynamic symbol table section */
#define SHT_INIT_ARRAY		14	/* Initialization function pointers. */
#define SHT_FINI_ARRAY		15	/* Termination function pointers. */
#define SHT_PREINIT_ARRAY	16	/* Pre-initialization function ptrs. */
#define SHT_GROUP		17	/* Section group. */
#define SHT_SYMTAB_SHNDX	18	/* Section indexes (see SHN_XINDEX). */
#define SHT_LOOS	0x60000000	/* First of OS specific semantics */
#define SHT_HIOS	0x6fffffff	/* Last of OS specific semantics */
#define SHT_GNU_VERDEF	0x6ffffffd
#define SHT_GNU_VERNEED	0x6ffffffe
#define SHT_GNU_VERSYM	0x6fffffff
#define SHT_LOPROC	0x70000000	/* reserved range for processor */
#define SHT_HIPROC	0x7fffffff	/* specific section header types */
#define SHT_LOUSER	0x80000000	/* reserved range for application */
#define SHT_HIUSER	0xffffffff	/* specific indexes */

/* Flags for sh_flags. */
#define SHF_WRITE		0x1	/* Section contains writable data. */
#define SHF_ALLOC		0x2	/* Section occupies memory. */
#define SHF_EXECINSTR		0x4	/* Section contains instructions. */
#define SHF_MERGE		0x10	/* Section may be merged. */
#define SHF_STRINGS		0x20	/* Section contains strings. */
#define SHF_INFO_LINK		0x40	/* sh_info holds section index. */
#define SHF_LINK_ORDER		0x80	/* Special ordering requirements. */
#define SHF_OS_NONCONFORMING	0x100	/* OS-specific processing required. */
#define SHF_GROUP		0x200	/* Member of section group. */
#define SHF_TLS			0x400	/* Section contains TLS data. */
#define SHF_MASKOS	0x0ff00000	/* OS-specific semantics. */
#define SHF_MASKPROC	0xf0000000	/* Processor-specific semantics. */

/* Values for p_type. */
#define PT_NULL		0	/* Unused entry. */
#define PT_LOAD		1	/* Loadable segment. */
#define PT_DYNAMIC	2	/* Dynamic linking information segment. */
#define PT_INTERP	3	/* Pathname of interpreter. */
#define PT_NOTE		4	/* Auxiliary information. */
#define PT_SHLIB	5	/* Reserved (not used). */
#define PT_PHDR		6	/* Location of program header itself. */
#define PT_TLS		7	/* Thread local storage segment */
#define PT_LOOS		0x60000000	/* First OS-specific. */
#define PT_HIOS		0x6fffffff	/* Last OS-specific. */
#define PT_LOPROC	0x70000000	/* First processor-specific type. */
#define PT_HIPROC	0x7fffffff	/* Last processor-specific type. */
#define PT_GNU_STACK	0x6474e551
#define PT_PAX_FLAGS	0x65041580

/* Values for p_flags. */
#define PF_X		0x1		/* Executable. */
#define PF_W		0x2		/* Writable. */
#define PF_R		0x4		/* Readable. */
#define PF_MASKOS	0x0ff00000	/* Operating system-specific. */
#define PF_MASKPROC	0xf0000000	/* Processor-specific. */

/* Values for d_tag. */
#define DT_NULL		0	/* Terminating entry. */
/* String table offset of a needed shared library. */
#define DT_NEEDED	1
#define DT_PLTRELSZ	2	/* Total size in bytes of PLT relocations. */
#define DT_PLTGOT	3	/* Processor-dependent address. */
#define DT_HASH		4	/* Address of symbol hash table. */
#define DT_STRTAB	5	/* Address of string table. */
#define DT_SYMTAB	6	/* Address of symbol table. */
#define DT_RELA		7	/* Address of ElfNN_Rela relocations. */
#define DT_RELASZ	8	/* Total size of ElfNN_Rela relocations. */
#define DT_RELAENT	9	/* Size of each ElfNN_Rela relocation entry. */
#define DT_STRSZ	10	/* Size of string table. */
#define DT_SYMENT	11	/* Size of each symbol table entry. */
#define DT_INIT		12	/* Address of initialization function. */
#define DT_FINI		13	/* Address of finalization function. */
/* String table offset of shared object name. */
#define DT_SONAME	14
#define DT_RPATH	15	/* String table offset of library path. [sup] */
#define DT_SYMBOLIC	16	/* Indicates "symbolic" linking. [sup] */
#define DT_REL		17	/* Address of ElfNN_Rel relocations. */
#define DT_RELSZ	18	/* Total size of ElfNN_Rel relocations. */
#define DT_RELENT	19	/* Size of each ElfNN_Rel relocation. */
#define DT_PLTREL	20	/* Type of relocation used for PLT. */
#define DT_DEBUG	21	/* Reserved (not used). */
/* Indicates there may be relocations in non-writable segments. [sup] */
#define DT_TEXTREL	22
#define DT_JMPREL	23	/* Address of PLT relocations. */
#define	DT_BIND_NOW	24	/* [sup] */
/* Address of the array of pointers to initialization functions */
#define	DT_INIT_ARRAY	25
/* Address of the array of pointers to termination functions */
#define	DT_FINI_ARRAY	26
/* Size in bytes of the array of initialization functions. */
#define	DT_INIT_ARRAYSZ	27
/* Size in bytes of the array of terminationfunctions. */
#define	DT_FINI_ARRAYSZ	28
/* String table offset of a null-terminated library search path string. */
#define	DT_RUNPATH	29
#define	DT_FLAGS	30	/* Object specific flag values. */
/*	Values greater than or equal to DT_ENCODING and less than
	DT_LOOS follow the rules for the interpretation of the d_un
	union as follows: even == 'd_ptr', even == 'd_val' or none */
#define	DT_ENCODING	32
/* Address of the array of pointers to pre-initialization functions. */
#define	DT_PREINIT_ARRAY 32
/* Size in bytes of the array of pre-initialization functions. */
#define	DT_PREINIT_ARRAYSZ 33
#define	DT_LOOS		0x6000000d	/* First OS-specific */
#define	DT_HIOS		0x6ffff000	/* Last OS-specific */
#define	DT_LOPROC	0x70000000	/* First processor-specific type. */
#define	DT_HIPROC	0x7fffffff	/* Last processor-specific type. */

#define	DT_VERNEED	0x6ffffffe
#define	DT_VERNEEDNUM	0x6fffffff
#define	DT_VERSYM	0x6ffffff0

/* Values for DT_FLAGS */
/*	Indicates that the object being loaded may make reference to
	the $ORIGIN substitution string */
#define	DF_ORIGIN	0x0001
#define	DF_SYMBOLIC	0x0002	/* Indicates "symbolic" linking. */
/* Indicates there may be relocations in non-writable segments. */
#define	DF_TEXTREL	0x0004
/*	Indicates that the dynamic linker should process all
	relocations for the object containing this entry before
	transferring control to the program.  */
#define	DF_BIND_NOW	0x0008
/*	Indicates that the shared object or executable contains code
	using a static thread-local storage scheme.  */
#define	DF_STATIC_TLS	0x0010

/* Values for n_type.  Used in core files. */
#define NT_PRSTATUS	1	/* Process status. */
#define NT_FPREGSET	2	/* Floating point registers. */
#define NT_PRPSINFO	3	/* Process state info. */

/* Symbol Binding - ELFNN_ST_BIND - st_info */
#define STB_LOCAL	0	/* Local symbol */
#define STB_GLOBAL	1	/* Global symbol */
#define STB_WEAK	2	/* like global - lower precedence */
#define STB_LOOS	10	/* Reserved range for operating system */
#define STB_HIOS	12	/*   specific semantics. */
#define STB_LOPROC	13	/* reserved range for processor */
#define STB_HIPROC	15	/*   specific semantics. */

/* Symbol type - ELFNN_ST_TYPE - st_info */
#define STT_NOTYPE	0	/* Unspecified type. */
#define STT_OBJECT	1	/* Data object. */
#define STT_FUNC	2	/* Function. */
#define STT_SECTION	3	/* Section. */
#define STT_FILE	4	/* Source file. */
#define STT_COMMON	5	/* Uninitialized common block. */
#define STT_TLS		6	/* TLS object. */
#define STT_LOOS	10	/* Reserved range for operating system */
#define STT_HIOS	12	/*   specific semantics. */
#define STT_LOPROC	13	/* reserved range for processor */
#define STT_HIPROC	15	/*   specific semantics. */

/* Symbol visibility - ELFNN_ST_VISIBILITY - st_other */
#define STV_DEFAULT	0x0	/* Default visibility (see binding). */
#define STV_INTERNAL	0x1	/* Special meaning in relocatable objects. */
#define STV_HIDDEN	0x2	/* Not visible. */
#define STV_PROTECTED	0x3	/* Visible but not preemptible. */

/* Special symbol table indexes. */
#define STN_UNDEF	0	/* Undefined symbol index. */

/*
 * ELF definitions common to all 32-bit architectures.
 */

typedef uint32	Elf32_Addr;
typedef uint16	Elf32_Half;
typedef uint32	Elf32_Off;
typedef int32		Elf32_Sword;
typedef uint32	Elf32_Word;

typedef Elf32_Word	Elf32_Hashelt;

/* Non-standard class-dependent datatype used for abstraction. */
typedef Elf32_Word	Elf32_Size;
typedef Elf32_Sword	Elf32_Ssize;

/*
 * ELF header.
 */

typedef struct {
	unsigned char	ident[EI_NIDENT];	/* File identification. */
	Elf32_Half	type;		/* File type. */
	Elf32_Half	machine;	/* Machine architecture. */
	Elf32_Word	version;	/* ELF format version. */
	Elf32_Addr	entry;	/* Entry point. */
	Elf32_Off	phoff;	/* Program header file offset. */
	Elf32_Off	shoff;	/* Section header file offset. */
	Elf32_Word	flags;	/* Architecture-specific flags. */
	Elf32_Half	ehsize;	/* Size of ELF header in bytes. */
	Elf32_Half	phentsize;	/* Size of program header entry. */
	Elf32_Half	phnum;	/* Number of program header entries. */
	Elf32_Half	shentsize;	/* Size of section header entry. */
	Elf32_Half	shnum;	/* Number of section header entries. */
	Elf32_Half	shstrndx;	/* Section name strings section. */
} Elf32_Ehdr;

/*
 * Section header.
 */

typedef struct {
	Elf32_Word	name;	/* Section name (index into the
					   section header string table). */
	Elf32_Word	type;	/* Section type. */
	Elf32_Word	flags;	/* Section flags. */
	Elf32_Addr	vaddr;	/* Address in memory image. */
	Elf32_Off	off;	/* Offset in file. */
	Elf32_Word	size;	/* Size in bytes. */
	Elf32_Word	link;	/* Index of a related section. */
	Elf32_Word	info;	/* Depends on section type. */
	Elf32_Word	addralign;	/* Alignment in bytes. */
	Elf32_Word	entsize;	/* Size of each entry in section. */
} Elf32_Shdr;

/*
 * Program header.
 */

typedef struct {
	Elf32_Word	type;		/* Entry type. */
	Elf32_Off	off;	/* File offset of contents. */
	Elf32_Addr	vaddr;	/* Virtual address in memory image. */
	Elf32_Addr	paddr;	/* Physical address (not used). */
	Elf32_Word	filesz;	/* Size of contents in file. */
	Elf32_Word	memsz;	/* Size of contents in memory. */
	Elf32_Word	flags;	/* Access permission flags. */
	Elf32_Word	align;	/* Alignment in memory and file. */
} Elf32_Phdr;

/*
 * Dynamic structure.  The ".dynamic" section contains an array of them.
 */

typedef struct {
	Elf32_Sword	d_tag;		/* Entry type. */
	union {
		Elf32_Word	d_val;	/* Integer value. */
		Elf32_Addr	d_ptr;	/* Address value. */
	} d_un;
} Elf32_Dyn;

/*
 * Relocation entries.
 */

/* Relocations that don't need an addend field. */
typedef struct {
	Elf32_Addr	off;	/* Location to be relocated. */
	Elf32_Word	info;		/* Relocation type and symbol index. */
} Elf32_Rel;

/* Relocations that need an addend field. */
typedef struct {
	Elf32_Addr	off;	/* Location to be relocated. */
	Elf32_Word	info;		/* Relocation type and symbol index. */
	Elf32_Sword	addend;	/* Addend. */
} Elf32_Rela;

/* Macros for accessing the fields of r_info. */
#define ELF32_R_SYM(info)	((info) >> 8)
#define ELF32_R_TYPE(info)	((unsigned char)(info))

/* Macro for constructing r_info from field values. */
#define ELF32_R_INFO(sym, type)	(((sym) << 8) + (unsigned char)(type))

/*
 * Relocation types.
 */

#define	R_X86_64_NONE	0	/* No relocation. */
#define	R_X86_64_64	1	/* Add 64 bit symbol value. */
#define	R_X86_64_PC32	2	/* PC-relative 32 bit signed sym value. */
#define	R_X86_64_GOT32	3	/* PC-relative 32 bit GOT offset. */
#define	R_X86_64_PLT32	4	/* PC-relative 32 bit PLT offset. */
#define	R_X86_64_COPY	5	/* Copy data from shared object. */
#define	R_X86_64_GLOB_DAT 6	/* Set GOT entry to data address. */
#define	R_X86_64_JMP_SLOT 7	/* Set GOT entry to code address. */
#define	R_X86_64_RELATIVE 8	/* Add load address of shared object. */
#define	R_X86_64_GOTPCREL 9	/* Add 32 bit signed pcrel offset to GOT. */
#define	R_X86_64_32	10	/* Add 32 bit zero extended symbol value */
#define	R_X86_64_32S	11	/* Add 32 bit sign extended symbol value */
#define	R_X86_64_16	12	/* Add 16 bit zero extended symbol value */
#define	R_X86_64_PC16	13	/* Add 16 bit signed extended pc relative symbol value */
#define	R_X86_64_8	14	/* Add 8 bit zero extended symbol value */
#define	R_X86_64_PC8	15	/* Add 8 bit signed extended pc relative symbol value */
#define	R_X86_64_DTPMOD64 16	/* ID of module containing symbol */
#define	R_X86_64_DTPOFF64 17	/* Offset in TLS block */
#define	R_X86_64_TPOFF64 18	/* Offset in static TLS block */
#define	R_X86_64_TLSGD	19	/* PC relative offset to GD GOT entry */
#define	R_X86_64_TLSLD	20	/* PC relative offset to LD GOT entry */
#define	R_X86_64_DTPOFF32 21	/* Offset in TLS block */
#define	R_X86_64_GOTTPOFF 22	/* PC relative offset to IE GOT entry */
#define	R_X86_64_TPOFF32 23	/* Offset in static TLS block */

#define	R_X86_64_COUNT	24	/* Count of defined relocation types. */


#define	R_ALPHA_NONE		0	/* No reloc */
#define	R_ALPHA_REFLONG		1	/* Direct 32 bit */
#define	R_ALPHA_REFQUAD		2	/* Direct 64 bit */
#define	R_ALPHA_GPREL32		3	/* GP relative 32 bit */
#define	R_ALPHA_LITERAL		4	/* GP relative 16 bit w/optimization */
#define	R_ALPHA_LITUSE		5	/* Optimization hint for LITERAL */
#define	R_ALPHA_GPDISP		6	/* Add displacement to GP */
#define	R_ALPHA_BRADDR		7	/* PC+4 relative 23 bit shifted */
#define	R_ALPHA_HINT		8	/* PC+4 relative 16 bit shifted */
#define	R_ALPHA_SREL16		9	/* PC relative 16 bit */
#define	R_ALPHA_SREL32		10	/* PC relative 32 bit */
#define	R_ALPHA_SREL64		11	/* PC relative 64 bit */
#define	R_ALPHA_OP_PUSH		12	/* OP stack push */
#define	R_ALPHA_OP_STORE	13	/* OP stack pop and store */
#define	R_ALPHA_OP_PSUB		14	/* OP stack subtract */
#define	R_ALPHA_OP_PRSHIFT	15	/* OP stack right shift */
#define	R_ALPHA_GPVALUE		16
#define	R_ALPHA_GPRELHIGH	17
#define	R_ALPHA_GPRELLOW	18
#define	R_ALPHA_IMMED_GP_16	19
#define	R_ALPHA_IMMED_GP_HI32	20
#define	R_ALPHA_IMMED_SCN_HI32	21
#define	R_ALPHA_IMMED_BR_HI32	22
#define	R_ALPHA_IMMED_LO32	23
#define	R_ALPHA_COPY		24	/* Copy symbol at runtime */
#define	R_ALPHA_GLOB_DAT	25	/* Create GOT entry */
#define	R_ALPHA_JMP_SLOT	26	/* Create PLT entry */
#define	R_ALPHA_RELATIVE	27	/* Adjust by program base */

#define	R_ALPHA_COUNT		28


#define	R_ARM_NONE		0	/* No relocation. */
#define	R_ARM_PC24		1
#define	R_ARM_ABS32		2
#define	R_ARM_REL32		3
#define	R_ARM_PC13		4
#define	R_ARM_ABS16		5
#define	R_ARM_ABS12		6
#define	R_ARM_THM_ABS5		7
#define	R_ARM_ABS8		8
#define	R_ARM_SBREL32		9
#define	R_ARM_THM_PC22		10
#define	R_ARM_THM_PC8		11
#define	R_ARM_AMP_VCALL9	12
#define	R_ARM_SWI24		13
#define	R_ARM_THM_SWI8		14
#define	R_ARM_XPC25		15
#define	R_ARM_THM_XPC22		16
#define	R_ARM_COPY		20	/* Copy data from shared object. */
#define	R_ARM_GLOB_DAT		21	/* Set GOT entry to data address. */
#define	R_ARM_JUMP_SLOT		22	/* Set GOT entry to code address. */
#define	R_ARM_RELATIVE		23	/* Add load address of shared object. */
#define	R_ARM_GOTOFF		24	/* Add GOT-relative symbol address. */
#define	R_ARM_GOTPC		25	/* Add PC-relative GOT table address. */
#define	R_ARM_GOT32		26	/* Add PC-relative GOT offset. */
#define	R_ARM_PLT32		27	/* Add PC-relative PLT offset. */
#define	R_ARM_CALL		28
#define	R_ARM_JUMP24	29
#define	R_ARM_V4BX		40
#define	R_ARM_GOT_PREL		96
#define	R_ARM_GNU_VTENTRY	100
#define	R_ARM_GNU_VTINHERIT	101
#define	R_ARM_TLS_IE32		107
#define	R_ARM_TLS_LE32		108
#define	R_ARM_RSBREL32		250
#define	R_ARM_THM_RPC22		251
#define	R_ARM_RREL32		252
#define	R_ARM_RABS32		253
#define	R_ARM_RPC24		254
#define	R_ARM_RBASE		255

#define	R_ARM_COUNT		38	/* Count of defined relocation types. */


#define	R_386_NONE	0	/* No relocation. */
#define	R_386_32	1	/* Add symbol value. */
#define	R_386_PC32	2	/* Add PC-relative symbol value. */
#define	R_386_GOT32	3	/* Add PC-relative GOT offset. */
#define	R_386_PLT32	4	/* Add PC-relative PLT offset. */
#define	R_386_COPY	5	/* Copy data from shared object. */
#define	R_386_GLOB_DAT	6	/* Set GOT entry to data address. */
#define	R_386_JMP_SLOT	7	/* Set GOT entry to code address. */
#define	R_386_RELATIVE	8	/* Add load address of shared object. */
#define	R_386_GOTOFF	9	/* Add GOT-relative symbol address. */
#define	R_386_GOTPC	10	/* Add PC-relative GOT table address. */
#define	R_386_TLS_TPOFF	14	/* Negative offset in static TLS block */
#define	R_386_TLS_IE	15	/* Absolute address of GOT for -ve static TLS */
#define	R_386_TLS_GOTIE	16	/* GOT entry for negative static TLS block */
#define	R_386_TLS_LE	17	/* Negative offset relative to static TLS */
#define	R_386_TLS_GD	18	/* 32 bit offset to GOT (index,off) pair */
#define	R_386_TLS_LDM	19	/* 32 bit offset to GOT (index,zero) pair */
#define	R_386_TLS_GD_32	24	/* 32 bit offset to GOT (index,off) pair */
#define	R_386_TLS_GD_PUSH 25	/* pushl instruction for Sun ABI GD sequence */
#define	R_386_TLS_GD_CALL 26	/* call instruction for Sun ABI GD sequence */
#define	R_386_TLS_GD_POP 27	/* popl instruction for Sun ABI GD sequence */
#define	R_386_TLS_LDM_32 28	/* 32 bit offset to GOT (index,zero) pair */
#define	R_386_TLS_LDM_PUSH 29	/* pushl instruction for Sun ABI LD sequence */
#define	R_386_TLS_LDM_CALL 30	/* call instruction for Sun ABI LD sequence */
#define	R_386_TLS_LDM_POP 31	/* popl instruction for Sun ABI LD sequence */
#define	R_386_TLS_LDO_32 32	/* 32 bit offset from start of TLS block */
#define	R_386_TLS_IE_32	33	/* 32 bit offset to GOT static TLS offset entry */
#define	R_386_TLS_LE_32	34	/* 32 bit offset within static TLS block */
#define	R_386_TLS_DTPMOD32 35	/* GOT entry containing TLS index */
#define	R_386_TLS_DTPOFF32 36	/* GOT entry containing TLS offset */
#define	R_386_TLS_TPOFF32 37	/* GOT entry of -ve static TLS offset */

#define	R_386_COUNT	38	/* Count of defined relocation types. */

#define	R_PPC_NONE		0	/* No relocation. */
#define	R_PPC_ADDR32		1
#define	R_PPC_ADDR24		2
#define	R_PPC_ADDR16		3
#define	R_PPC_ADDR16_LO		4
#define	R_PPC_ADDR16_HI		5
#define	R_PPC_ADDR16_HA		6
#define	R_PPC_ADDR14		7
#define	R_PPC_ADDR14_BRTAKEN	8
#define	R_PPC_ADDR14_BRNTAKEN	9
#define	R_PPC_REL24		10
#define	R_PPC_REL14		11
#define	R_PPC_REL14_BRTAKEN	12
#define	R_PPC_REL14_BRNTAKEN	13
#define	R_PPC_GOT16		14
#define	R_PPC_GOT16_LO		15
#define	R_PPC_GOT16_HI		16
#define	R_PPC_GOT16_HA		17
#define	R_PPC_PLTREL24		18
#define	R_PPC_COPY		19
#define	R_PPC_GLOB_DAT		20
#define	R_PPC_JMP_SLOT		21
#define	R_PPC_RELATIVE		22
#define	R_PPC_LOCAL24PC		23
#define	R_PPC_UADDR32		24
#define	R_PPC_UADDR16		25
#define	R_PPC_REL32		26
#define	R_PPC_PLT32		27
#define	R_PPC_PLTREL32		28
#define	R_PPC_PLT16_LO		29
#define	R_PPC_PLT16_HI		30
#define	R_PPC_PLT16_HA		31
#define	R_PPC_SDAREL16		32
#define	R_PPC_SECTOFF		33
#define	R_PPC_SECTOFF_LO	34
#define	R_PPC_SECTOFF_HI	35
#define	R_PPC_SECTOFF_HA	36

#define	R_PPC_COUNT		37	/* Count of defined relocation types. */

#define R_PPC_TLS		67
#define R_PPC_DTPMOD32		68
#define R_PPC_TPREL16		69
#define R_PPC_TPREL16_LO	70
#define R_PPC_TPREL16_HI	71
#define R_PPC_TPREL16_HA	72
#define R_PPC_TPREL32		73
#define R_PPC_DTPREL16		74
#define R_PPC_DTPREL16_LO	75
#define R_PPC_DTPREL16_HI	76
#define R_PPC_DTPREL16_HA	77
#define R_PPC_DTPREL32		78
#define R_PPC_GOT_TLSGD16	79
#define R_PPC_GOT_TLSGD16_LO	80
#define R_PPC_GOT_TLSGD16_HI	81
#define R_PPC_GOT_TLSGD16_HA	82
#define R_PPC_GOT_TLSLD16	83
#define R_PPC_GOT_TLSLD16_LO	84
#define R_PPC_GOT_TLSLD16_HI	85
#define R_PPC_GOT_TLSLD16_HA	86
#define R_PPC_GOT_TPREL16	87
#define R_PPC_GOT_TPREL16_LO	88
#define R_PPC_GOT_TPREL16_HI	89
#define R_PPC_GOT_TPREL16_HA	90

#define	R_PPC_EMB_NADDR32	101
#define	R_PPC_EMB_NADDR16	102
#define	R_PPC_EMB_NADDR16_LO	103
#define	R_PPC_EMB_NADDR16_HI	104
#define	R_PPC_EMB_NADDR16_HA	105
#define	R_PPC_EMB_SDAI16	106
#define	R_PPC_EMB_SDA2I16	107
#define	R_PPC_EMB_SDA2REL	108
#define	R_PPC_EMB_SDA21		109
#define	R_PPC_EMB_MRKREF	110
#define	R_PPC_EMB_RELSEC16	111
#define	R_PPC_EMB_RELST_LO	112
#define	R_PPC_EMB_RELST_HI	113
#define	R_PPC_EMB_RELST_HA	114
#define	R_PPC_EMB_BIT_FLD	115
#define	R_PPC_EMB_RELSDA	116

					/* Count of defined relocation types. */
#define	R_PPC_EMB_COUNT		(R_PPC_EMB_RELSDA - R_PPC_EMB_NADDR32 + 1)


#define R_SPARC_NONE		0
#define R_SPARC_8		1
#define R_SPARC_16		2
#define R_SPARC_32		3
#define R_SPARC_DISP8		4
#define R_SPARC_DISP16		5
#define R_SPARC_DISP32		6
#define R_SPARC_WDISP30		7
#define R_SPARC_WDISP22		8
#define R_SPARC_HI22		9
#define R_SPARC_22		10
#define R_SPARC_13		11
#define R_SPARC_LO10		12
#define R_SPARC_GOT10		13
#define R_SPARC_GOT13		14
#define R_SPARC_GOT22		15
#define R_SPARC_PC10		16
#define R_SPARC_PC22		17
#define R_SPARC_WPLT30		18
#define R_SPARC_COPY		19
#define R_SPARC_GLOB_DAT	20
#define R_SPARC_JMP_SLOT	21
#define R_SPARC_RELATIVE	22
#define R_SPARC_UA32		23
#define R_SPARC_PLT32		24
#define R_SPARC_HIPLT22		25
#define R_SPARC_LOPLT10		26
#define R_SPARC_PCPLT32		27
#define R_SPARC_PCPLT22		28
#define R_SPARC_PCPLT10		29
#define R_SPARC_10		30
#define R_SPARC_11		31
#define R_SPARC_64		32
#define R_SPARC_OLO10		33
#define R_SPARC_HH22		34
#define R_SPARC_HM10		35
#define R_SPARC_LM22		36
#define R_SPARC_PC_HH22		37
#define R_SPARC_PC_HM10		38
#define R_SPARC_PC_LM22		39
#define R_SPARC_WDISP16		40
#define R_SPARC_WDISP19		41
#define R_SPARC_GLOB_JMP	42
#define R_SPARC_7		43
#define R_SPARC_5		44
#define R_SPARC_6		45
#define	R_SPARC_DISP64		46
#define	R_SPARC_PLT64		47
#define	R_SPARC_HIX22		48
#define	R_SPARC_LOX10		49
#define	R_SPARC_H44		50
#define	R_SPARC_M44		51
#define	R_SPARC_L44		52
#define	R_SPARC_REGISTER	53
#define	R_SPARC_UA64		54
#define	R_SPARC_UA16		55


/*
 * Magic number for the elf trampoline, chosen wisely to be an immediate
 * value.
 */
#define ARM_MAGIC_TRAMP_NUMBER	0x5c000003


/*
 * Symbol table entries.
 */

typedef struct {
	Elf32_Word	name;	/* String table index of name. */
	Elf32_Addr	value;	/* Symbol value. */
	Elf32_Word	size;	/* Size of associated object. */
	unsigned char	info;	/* Type and binding information. */
	unsigned char	other;	/* Reserved (not used). */
	Elf32_Half	shndx;	/* Section index of symbol. */
} Elf32_Sym;

/* Macros for accessing the fields of st_info. */
#define ELF32_ST_BIND(info)		((info) >> 4)
#define ELF32_ST_TYPE(info)		((info) & 0xf)

/* Macro for constructing st_info from field values. */
#define ELF32_ST_INFO(bind, type)	(((bind) << 4) + ((type) & 0xf))

/* Macro for accessing the fields of st_other. */
#define ELF32_ST_VISIBILITY(oth)	((oth) & 0x3)

/*
 * ELF definitions common to all 64-bit architectures.
 */

typedef uint64	Elf64_Addr;
typedef uint16	Elf64_Half;
typedef uint64	Elf64_Off;
typedef int32		Elf64_Sword;
typedef int64		Elf64_Sxword;
typedef uint32	Elf64_Word;
typedef uint64	Elf64_Xword;

/*
 * Types of dynamic symbol hash table bucket and chain elements.
 *
 * This is inconsistent among 64 bit architectures, so a machine dependent
 * typedef is required.
 */

#ifdef __alpha__
typedef Elf64_Off	Elf64_Hashelt;
#else
typedef Elf64_Word	Elf64_Hashelt;
#endif

/* Non-standard class-dependent datatype used for abstraction. */
typedef Elf64_Xword	Elf64_Size;
typedef Elf64_Sxword	Elf64_Ssize;

/*
 * ELF header.
 */

typedef struct {
	unsigned char	ident[EI_NIDENT];	/* File identification. */
	Elf64_Half	type;		/* File type. */
	Elf64_Half	machine;	/* Machine architecture. */
	Elf64_Word	version;	/* ELF format version. */
	Elf64_Addr	entry;	/* Entry point. */
	Elf64_Off	phoff;	/* Program header file offset. */
	Elf64_Off	shoff;	/* Section header file offset. */
	Elf64_Word	flags;	/* Architecture-specific flags. */
	Elf64_Half	ehsize;	/* Size of ELF header in bytes. */
	Elf64_Half	phentsize;	/* Size of program header entry. */
	Elf64_Half	phnum;	/* Number of program header entries. */
	Elf64_Half	shentsize;	/* Size of section header entry. */
	Elf64_Half	shnum;	/* Number of section header entries. */
	Elf64_Half	shstrndx;	/* Section name strings section. */
} Elf64_Ehdr;

/*
 * Section header.
 */

typedef struct Elf64_Shdr Elf64_Shdr;
struct Elf64_Shdr {
	Elf64_Word	name;	/* Section name (index into the
					   section header string table). */
	Elf64_Word	type;	/* Section type. */
	Elf64_Xword	flags;	/* Section flags. */
	Elf64_Addr	addr;	/* Address in memory image. */
	Elf64_Off	off;	/* Offset in file. */
	Elf64_Xword	size;	/* Size in bytes. */
	Elf64_Word	link;	/* Index of a related section. */
	Elf64_Word	info;	/* Depends on section type. */
	Elf64_Xword	addralign;	/* Alignment in bytes. */
	Elf64_Xword	entsize;	/* Size of each entry in section. */
	
	int	shnum;  /* section number, not stored on disk */
	LSym*	secsym; /* section symbol, if needed; not on disk */
};

/*
 * Program header.
 */

typedef struct {
	Elf64_Word	type;		/* Entry type. */
	Elf64_Word	flags;	/* Access permission flags. */
	Elf64_Off	off;	/* File offset of contents. */
	Elf64_Addr	vaddr;	/* Virtual address in memory image. */
	Elf64_Addr	paddr;	/* Physical address (not used). */
	Elf64_Xword	filesz;	/* Size of contents in file. */
	Elf64_Xword	memsz;	/* Size of contents in memory. */
	Elf64_Xword	align;	/* Alignment in memory and file. */
} Elf64_Phdr;

/*
 * Dynamic structure.  The ".dynamic" section contains an array of them.
 */

typedef struct {
	Elf64_Sxword	d_tag;		/* Entry type. */
	union {
		Elf64_Xword	d_val;	/* Integer value. */
		Elf64_Addr	d_ptr;	/* Address value. */
	} d_un;
} Elf64_Dyn;

/*
 * Relocation entries.
 */

/* Relocations that don't need an addend field. */
typedef struct {
	Elf64_Addr	off;	/* Location to be relocated. */
	Elf64_Xword	info;		/* Relocation type and symbol index. */
} Elf64_Rel;

/* Relocations that need an addend field. */
typedef struct {
	Elf64_Addr	off;	/* Location to be relocated. */
	Elf64_Xword	info;		/* Relocation type and symbol index. */
	Elf64_Sxword	addend;	/* Addend. */
} Elf64_Rela;

/* Macros for accessing the fields of r_info. */
#define ELF64_R_SYM(info)	((info) >> 32)
#define ELF64_R_TYPE(info)	((info) & 0xffffffffL)

/* Macro for constructing r_info from field values. */
#define ELF64_R_INFO(sym, type)	((((uint64)(sym)) << 32) + (((uint64)(type)) & 0xffffffffULL))

/*
 * Symbol table entries.
 */

typedef struct {
	Elf64_Word	name;	/* String table index of name. */
	unsigned char	info;	/* Type and binding information. */
	unsigned char	other;	/* Reserved (not used). */
	Elf64_Half	shndx;	/* Section index of symbol. */
	Elf64_Addr	value;	/* Symbol value. */
	Elf64_Xword	size;	/* Size of associated object. */
} Elf64_Sym;

/* Macros for accessing the fields of st_info. */
#define ELF64_ST_BIND(info)		((info) >> 4)
#define ELF64_ST_TYPE(info)		((info) & 0xf)

/* Macro for constructing st_info from field values. */
#define ELF64_ST_INFO(bind, type)	(((bind) << 4) + ((type) & 0xf))

/* Macro for accessing the fields of st_other. */
#define ELF64_ST_VISIBILITY(oth)	((oth) & 0x3)

/*
 * Go linker interface
 */

#define	ELF64HDRSIZE	64
#define	ELF64PHDRSIZE	56
#define	ELF64SHDRSIZE	64
#define	ELF64RELSIZE	16
#define	ELF64RELASIZE	24
#define	ELF64SYMSIZE	sizeof(Elf64_Sym)

#define	ELF32HDRSIZE	sizeof(Elf32_Ehdr)
#define	ELF32PHDRSIZE	sizeof(Elf32_Phdr)
#define	ELF32SHDRSIZE	sizeof(Elf32_Shdr)
#define	ELF32SYMSIZE	sizeof(Elf32_Sym)
#define	ELF32RELSIZE	8

/*
 * The interface uses the 64-bit structures always,
 * to avoid code duplication.  The writers know how to
 * marshal a 32-bit representation from the 64-bit structure.
 */
typedef Elf64_Ehdr ElfEhdr;
typedef Elf64_Shdr ElfShdr;
typedef Elf64_Phdr ElfPhdr;

void	elfinit(void);
ElfEhdr	*getElfEhdr(void);
ElfShdr	*newElfShdr(vlong);
ElfPhdr	*newElfPhdr(void);
uint32	elfwritehdr(void);
uint32	elfwritephdrs(void);
uint32	elfwriteshdrs(void);
void	elfwritedynent(LSym*, int, uint64);
void	elfwritedynentsym(LSym*, int, LSym*);
void	elfwritedynentsymsize(LSym*, int, LSym*);
uint32	elfhash(uchar*);
uint64	startelf(void);
uint64	endelf(void);
extern	int	numelfphdr;
extern	int	numelfshdr;
extern	int	iself;
extern	int	elfverneed;
int	elfinterp(ElfShdr*, uint64, uint64, char*);
int	elfwriteinterp(void);
int	elfnetbsdsig(ElfShdr*, uint64, uint64);
int	elfwritenetbsdsig(void);
int	elfopenbsdsig(ElfShdr*, uint64, uint64);
int	elfwriteopenbsdsig(void);
void	addbuildinfo(char*);
int	elfbuildinfo(ElfShdr*, uint64, uint64);
int	elfwritebuildinfo(void);
void	elfdynhash(void);
ElfPhdr* elfphload(Segment*);
ElfShdr* elfshbits(Section*);
ElfShdr* elfshalloc(Section*);
ElfShdr* elfshname(char*);
ElfShdr* elfshreloc(Section*);
void	elfsetstring(char*, int);
void	elfaddverneed(LSym*);
void	elfemitreloc(void);
void	shsym(ElfShdr*, LSym*);
void	phsh(ElfPhdr*, ElfShdr*);
void	doelf(void);
void	elfsetupplt(void);
void	dwarfaddshstrings(LSym*);
void	dwarfaddelfsectionsyms(void);
void	dwarfaddelfheaders(void);
void	asmbelf(vlong symo);
void	asmbelfsetup(void);
extern char linuxdynld[];
extern char freebsddynld[];
extern char netbsddynld[];
extern char openbsddynld[];
extern char dragonflydynld[];
extern char solarisdynld[];
int	elfreloc1(Reloc*, vlong sectoff);
void	putelfsectionsyms(void);

EXTERN	int	elfstrsize;
EXTERN	char*	elfstrdat;
EXTERN	int	buildinfolen;

/*
 * Total amount of space to reserve at the start of the file
 * for Header, PHeaders, SHeaders, and interp.
 * May waste some.
 * On FreeBSD, cannot be larger than a page.
 */
#define	ELFRESERVE	3072
