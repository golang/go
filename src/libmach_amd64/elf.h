// Inferno libmach/elf.h
// http://code.google.com/p/inferno-os/source/browse/utils/libmach/elf.h
//
// 	Copyright © 1994-1999 Lucent Technologies Inc.
// 	Power PC support Copyright © 1995-2004 C H Forsyth (forsyth@terzarima.net).
// 	Portions Copyright © 1997-1999 Vita Nuova Limited.
// 	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com).
// 	Revisions Copyright © 2000-2004 Lucent Technologies Inc. and others.
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

/*
 *	Definitions needed for  accessing ELF headers.
 *	32-bit and 64-bit structs differ.
 */
typedef struct {
	uchar	ident[16];	/* ident bytes */
	ushort	type;		/* file type */
	ushort	machine;	/* target machine */
	int	version;	/* file version */
	ulong	elfentry;	/* start address */
	ulong	phoff;		/* phdr file offset */
	ulong	shoff;		/* shdr file offset */
	int	flags;		/* file flags */
	ushort	ehsize;		/* sizeof ehdr */
	ushort	phentsize;	/* sizeof phdr */
	ushort	phnum;		/* number phdrs */
	ushort	shentsize;	/* sizeof shdr */
	ushort	shnum;		/* number shdrs */
	ushort	shstrndx;	/* shdr string index */
} Ehdr32;

typedef struct {
	uchar	ident[16];	/* ident bytes */
	ushort	type;		/* file type */
	ushort	machine;	/* target machine */
	int	version;	/* file version */
	uvlong	elfentry;	/* start address */
	uvlong	phoff;		/* phdr file offset */
	uvlong	shoff;		/* shdr file offset */
	int	flags;		/* file flags */
	ushort	ehsize;		/* sizeof ehdr */
	ushort	phentsize;	/* sizeof phdr */
	ushort	phnum;		/* number phdrs */
	ushort	shentsize;	/* sizeof shdr */
	ushort	shnum;		/* number shdrs */
	ushort	shstrndx;	/* shdr string index */
} Ehdr64;

typedef struct {
	int	type;		/* entry type */
	ulong	offset;		/* file offset */
	ulong	vaddr;		/* virtual address */
	ulong	paddr;		/* physical address */
	int	filesz;		/* file size */
	ulong	memsz;		/* memory size */
	int	flags;		/* entry flags */
	int	align;		/* memory/file alignment */
} Phdr32;

typedef struct {
	int	type;		/* entry type */
	int	flags;		/* entry flags */
	uvlong	offset;		/* file offset */
	uvlong	vaddr;		/* virtual address */
	uvlong	paddr;		/* physical address */
	uvlong	filesz;		/* file size */
	uvlong	memsz;		/* memory size */
	uvlong	align;		/* memory/file alignment */
} Phdr64;

typedef struct {
	ulong	name;		/* section name */
	ulong	type;		/* SHT_... */
	ulong	flags;		/* SHF_... */
	ulong	addr;		/* virtual address */
	ulong	offset;		/* file offset */
	ulong	size;		/* section size */
	ulong	link;		/* misc info */
	ulong	info;		/* misc info */
	ulong	addralign;	/* memory alignment */
	ulong	entsize;	/* entry size if table */
} Shdr32;

typedef struct {
	ulong	name;		/* section name */
	ulong	type;		/* SHT_... */
	uvlong	flags;		/* SHF_... */
	uvlong	addr;		/* virtual address */
	uvlong	offset;		/* file offset */
	uvlong	size;		/* section size */
	ulong	link;		/* misc info */
	ulong	info;		/* misc info */
	uvlong	addralign;	/* memory alignment */
	uvlong	entsize;	/* entry size if table */
} Shdr64;

enum {
	/* Ehdr codes */
	MAG0 = 0,		/* ident[] indexes */
	MAG1 = 1,
	MAG2 = 2,
	MAG3 = 3,
	CLASS = 4,
	DATA = 5,
	VERSION = 6,

	ELFCLASSNONE = 0,	/* ident[CLASS] */
	ELFCLASS32 = 1,
	ELFCLASS64 = 2,
	ELFCLASSNUM = 3,

	ELFDATANONE = 0,	/* ident[DATA] */
	ELFDATA2LSB = 1,
	ELFDATA2MSB = 2,
	ELFDATANUM = 3,

	NOETYPE = 0,		/* type */
	REL = 1,
	EXEC = 2,
	DYN = 3,
	CORE = 4,

	NONE = 0,		/* machine */
	M32 = 1,		/* AT&T WE 32100 */
	SPARC = 2,		/* Sun SPARC */
	I386 = 3,		/* Intel 80386 */
	M68K = 4,		/* Motorola 68000 */
	M88K = 5,		/* Motorola 88000 */
	I486 = 6,		/* Intel 80486 */
	I860 = 7,		/* Intel i860 */
	MIPS = 8,		/* Mips R2000 */
	S370 = 9,		/* Amdhal	*/
	SPARC64 = 18,		/* Sun SPARC v9 */
	POWER = 20,		/* PowerPC */
	ARM = 40,			/* ARM */
	AMD64 = 62,		/* Amd64 */

	NO_VERSION = 0,		/* version, ident[VERSION] */
	CURRENT = 1,

	/* Phdr Codes */
	NOPTYPE = 0,		/* type */
	LOAD = 1,
	DYNAMIC = 2,
	INTERP = 3,
	NOTE = 4,
	SHLIB = 5,
	PHDR = 6,

	R = 0x4,		/* flags */
	W = 0x2,
	X = 0x1,

	/* Shdr Codes */
	Progbits = 1,	/* section types */
	Strtab = 3,
	Nobits = 8,

	Swrite = 1,	/* section attributes */
	Salloc = 2,
	Sexec = 4,
};

#define	ELF_MAG		((0x7f<<24) | ('E'<<16) | ('L'<<8) | 'F')
