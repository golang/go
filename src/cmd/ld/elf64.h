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

typedef uint64	Elf64_Addr;	/* Unsigned program address */
typedef uint64	Elf64_Off;	/* Unsigned file offset */
typedef uint16	Elf64_Half;	/* Unsigned medium integer */
typedef uint32	Elf64_Word;	/* Unsigned integer */
typedef int32	Elf64_Sword;	/* Signed integer */
typedef uint64	Elf64_Xword;	/* Unsigned long integer */
typedef int64	Elf64_Sxword; 	/* Signed long integer */

typedef struct Elf64Hdr		Elf64Hdr;
typedef struct Elf64SHdr	Elf64SHdr;
typedef struct Elf64PHdr	Elf64PHdr;

#define	EI_NIDENT	16
struct Elf64Hdr
{
	uchar ident[EI_NIDENT];	/* ELF identification */
	Elf64_Half	type;	/* Object file type */
	Elf64_Half	machine;	/* Machine type */
	Elf64_Word	version;	/* Object file version */
	Elf64_Addr	entry;	/* Entry point address */
	Elf64_Off	phoff;	/* Program header offset */
	Elf64_Off	shoff;	/* Section header offset */
	Elf64_Word	flags;	/* Processor-specific flags */
	Elf64_Half	ehsize;	/* ELF header size */
	Elf64_Half	phentsize;	/* Size of program header entry */
	Elf64_Half	phnum;	/* Number of program header entries */
	Elf64_Half	shentsize;	/* Size of section header entry */
	Elf64_Half	shnum;	/* Number of section header entries */
	Elf64_Half	shstrndx;	/* Section name string table index */
};
#define	ELF64HDRSIZE	64

/* E ident indexes */
#define	EI_MAG0	0 	/* File identification */
#define	EI_MAG1	1
#define	EI_MAG2	2
#define	EI_MAG3	3
#define	EI_CLASS	4	/* File class */
#define	EI_DATA		5	/* Data encoding */
#define	EI_VERSION	6	/* File version */
#define	EI_OSABI	7	/* OS/ABI identification */
#define	EI_ABIVERSION	8	/* ABI version */
#define	EI_PAD	9	/*Start of padding bytes */

/* E types */
#define	ET_NONE	0	/* No file type */
#define	ET_REL	1	/* Relocatable object file */
#define	ET_EXEC	2	/* Executable file */
#define	ET_DYN	3	/* Shared object file */
#define	ET_CORE	4	/* Core file */
#define	ET_LOOS 0xFE00	/* Environment-specific use */
#define	ET_HIOS 0xFEFF
#define	ET_LOPROC 0xFF00	/* Processor-specific use */
#define	ET_HIPROC 0xFFFF

/* E classes */
#define	ELFCLASS32	1 	/* 32-bit objects */
#define	ELFCLASS64	2	/* 64-bit objects */

/* E endians */
#define	ELFDATA2LSB	1	/* little-endian */
#define	ELFDATA2MSB	2	/* big-endian */

#define	EV_CURRENT	1	/* current version of format */

struct Elf64PHdr
{
	Elf64_Word	type;	/* Type of segment */
	Elf64_Word	flags;	/* Segment attributes */
	Elf64_Off	off;	/* Offset in file */
	Elf64_Addr	vaddr;	/* Virtual address in memory */
	Elf64_Addr	paddr;	/* Reserved */
	Elf64_Xword	filesz;	/* Size of segment in file */
	Elf64_Xword	memsz;	/* Size of segment in memory */
	Elf64_Xword	align;	/* Alignment of segment */
};
#define	ELF64PHDRSIZE	56

/* P types */
#define	PT_NULL		0	/* Unused entry */
#define	PT_LOAD		1	/* Loadable segment */
#define	PT_DYNAMIC	2	/* Dynamic linking tables */
#define	PT_INTERP	3	/* Program interpreter path name */
#define	PT_NOTE		4	/* Note sections */
#define	PT_PHDR		6	/* Program header table */

/* P flags */
#define	PF_X	0x1	/* Execute permission */
#define	PF_W	0x2	/* Write permission */
#define	PF_R	0x4	/* Read permission */
#define	PF_MASKOS	0x00FF0000 /* reserved for environment-specific use */
#define	PF_MASKPROC	0xFF000000 /*reserved for processor-specific use */

struct Elf64SHdr
{
	Elf64_Word	name;	/* Section name */
	Elf64_Word	type;	/* Section type */
	Elf64_Xword	flags;	/* Section attributes */
	Elf64_Addr	addr;	/* Virtual address in memory */
	Elf64_Off	off;	/* Offset in file */
	Elf64_Xword	size;	/* Size of section */
	Elf64_Word	link;	/* Link to other section */
	Elf64_Word	info;	/* Miscellaneous information */
	Elf64_Xword	addralign;	/* Address alignment boundary */
	Elf64_Xword	entsize;	/* Size of entries, if section has table */
};
#define	ELF64SHDRSIZE	64

/* S types */
#define SHT_NULL	0	/* Unused section header */
#define SHT_PROGBITS	1	/* Information defined by the program */
#define SHT_SYMTAB	2	/* Linker symbol table */
#define SHT_STRTAB	3	/* String table */
#define SHT_RELA	4 	/* "Rela" type relocation entries */
#define SHT_HASH	5	/* Symbol hash table */
#define SHT_DYNAMIC	6	/* Dynamic linking tables */
#define SHT_NOTE	7	/* Note information */
#define SHT_NOBITS	8	/* Uninitialized space; does not occupy any space in the file */
#define SHT_REL		9	/* "Rel" type relocation entries */
#define SHT_SHLIB	10	/* Reserved */
#define SHT_DYNSYM	11	/* A dynamic loader symbol table */
#define SHT_LOOS	0x60000000	/* Environment-specific use */
#define SHT_HIOS	0x6FFFFFFF
#define SHT_LOPROC	0x70000000	/* Processor-specific use */
#define SHT_HIPROC 0x7FFFFFFF

/* S flags */
#define	SHF_WRITE	0x1 /* Writable data */
#define	SHF_ALLOC	0x2 /* Allocated in memory image of program */
#define	SHF_EXECINSTR	0x4 /* Executable instructions */
#define	SHF_MASKOS	0x0F000000	/* Environment-specific use */
#define	SHF_MASKPROC	0xF0000000	/* Processor-specific use */


typedef struct Elf64Dyn Elf64Dyn;
struct Elf64Dyn
{
	Elf64_Sxword d_tag;
	union {
		Elf64_Xword d_val;
		Elf64_Addr d_ptr;
	} d_un;
};

/* Dyn table entries */
#define	DT_NULL		0	/* ignored: Marks the end of the dynamic array */
#define	DT_NEEDED	1	/* d_val: The string table offset of the name of
				   a needed library. */
#define	DT_PLTRELSZ	2	/* d_val: Total size, in bytes, of the relocation
				   entries associated with  the procedure linkage table. */
#define	DT_PLTGOT	3	/* d_ptr: Contains an address associated with the linkage
				   table. The specific meaning of this field is
				   processor-dependent. */
#define	DT_HASH		4	/* d_ptr: Address of the symbol hash table. */
#define	DT_STRTAB	5	/* d_ptr: Address of the dynamic string table. */
#define	DT_SYMTAB	6	/* d_ptr: Address of the dynamic symbol table. */
#define	DT_RELA		7	/* d_ptr Address of a relocation table with Elf64_Rela
				   entries. */
#define	DT_RELASZ	8	/* d_val: Total size, in bytes, of the DT_RELA relocation
				   table. */
#define	DT_RELAENT	9	/* d_val: Size, in bytes, of each DT_RELA relocation
				   entry. */
#define	DT_STRSZ	10	/* d_val: Total size, in bytes, of the string table. */
#define	DT_SYMENT	11	/* d_val: Size, in bytes, of each symbol table entry. */
#define	DT_INIT		12	/* d_ptr Address of the initialization function. */
#define	DT_FINI		13	/* d_ptr Address of the termination function. */
#define	DT_SONAME	14	/* d_val The string table offset of the name of this
				   shared object. */
#define	DT_RPATH	15	/* d_val The string table offset of a shared library
				   search path string. */
#define	DT_SYMBOLIC	16	/* ignored The presence of this dynamic table entry
				   modifies the symbol resolution algorithm for references
				   within the library. Symbols defined within the library
				   are used to resolve references before the dynamic
				   linker searches the usual search path. */
#define	DT_REL		17	/* d_ptr Address of a relocation table with Elf64_Rel
				   entries. */
#define	DT_RELSZ	18	/* d_val Total size, in bytes, of the DT_REL relocation
				   table. */
#define	DT_RELENT	19	/* d_val Size, in bytes, of each DT_REL relocation
				   entry. */
#define	DT_PLTREL	20	/* d_val Type of relocation entry used for the procedure
				   linkage table. The d_val member contains either DT_REL
				   or DT_RELA. */
#define	DT_DEBUG	21	/* d_ptr Reserved for debugger use. */
#define	DT_TEXTREL	22	/* ignored The presence of this dynamic table entry
				   signals that the relocation table contains relocations
				   for a non-writable segment. */
#define	DT_JMPREL	23	/* d_ptr Address of the relocations associated with the
				   procedure linkage table. */
#define	DT_BIND_NOW	24	/* ignored The presence of this dynamic table entry
				   signals that the dynamic loader should process all
				   relocations for this object before transferring
				   control to the program. */
#define	DT_INIT_ARRAY	25	/* d_ptr Pointer to an array of pointers to initialization
				   functions. */
#define	DT_FINI_ARRAY	26	/* d_ptr Pointer to an array of pointers to termination
				   functions. */
#define	DT_INIT_ARRAYSZ	27	/* d_val Size, in bytes, of the array of initialization
				   functions. */
#define	DT_FINI_ARRAYSZ	28	/* d_val Size, in bytes, of the array of termination
				   functions. */
#define	DT_LOOS	0x60000000	/* Defines a range of dynamic table tags that are reserved
				   for environment-specific use. */
#define	DT_HIOS		0x6FFFFFFF
#define	DT_LOPROC	0x70000000	/* Defines a range of dynamic table tags that are
					   reserved for processor-specific use. */
#define	DT_HIPROC	0x7FFFFFFF

typedef struct Elf64_Rel Elf64_Rel;
struct Elf64_Rel
{
	Elf64_Addr r_offset;	/* Address of reference */
	Elf64_Xword r_info;	/* Symbol index and type of relocation */
};
#define ELF64RELSIZE 8

typedef struct Elf64_Rela Elf64_Rela;
struct Elf64_Rela
{
	Elf64_Addr r_offset;	/* Address of reference */
	Elf64_Xword r_info;	/* Symbol index and type of relocation */
	Elf64_Sxword r_addend;	/* Constant part of expression */
};
#define ELF64RELASIZE 24

#define	ELF64_R_SYM(i)	((i) >> 32)
#define	ELF64_R_TYPE(i)	((i) & 0xffffffffL)
#define	ELF64_R_INFO(s, t)	(((s) << 32) + ((t) & 0xffffffffL))

void	elf64init(void);
Elf64Hdr	*getElf64Hdr();
Elf64SHdr	*newElf64SHdr(char*);
Elf64PHdr	*newElf64PHdr();
uint32	elf64writehdr(void);
uint32	elf64writephdrs(void);
uint32	elf64writeshdrs(void);
void	elf64writestrtable(void);
void	elf64writedynent(int, uint64);
uint32	elf64_hash(uchar*);
uint64	startelf(void);
uint64	endelf(void);
extern	int	nume64phdr;
extern	int	nume64shdr;

#define	STRTABSIZE	256
/* Amount of space available for Header, PHeaders and SHeaders */
#define	ELF64FULLHDRSIZE	2048
/* Space reserved after ELF64FULLHEADERSIZE for dynamic info */
#define	ELFDYNAMICSIZE	256
/* Total amount of ELF space to reserve at the start of the file; may waste some */
#define	ELF64RESERVE	4096
