// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "../ld/elf.h"

/*
 * We use the 64-bit data structures on both 32- and 64-bit machines
 * in order to write the code just once.  The 64-bit data structure is
 * written in the 32-bit format on the 32-bit machines.
 */
#define	NSECT	16

static	int	elf64;
static	Elf64_Ehdr	hdr;
static	Elf64_Phdr	*phdr[NSECT];
static	Elf64_Shdr	*shdr[NSECT];
static	char	*sname[NSECT];

/*
 Initialize the global variable that describes the ELF header. It will be updated as
 we write section and prog headers.
 */
void
elfinit(void)
{
	switch(thechar) {
	// 64-bit architectures
	case '6':
		elf64 = 1;
		hdr.phoff = ELF64HDRSIZE;	/* Must be be ELF64HDRSIZE: first PHdr must follow ELF header */
		hdr.shoff = ELF64HDRSIZE;	/* Will move as we add PHeaders */
		hdr.ehsize = ELF64HDRSIZE;	/* Must be ELF64HDRSIZE */
		hdr.phentsize = ELF64PHDRSIZE;	/* Must be ELF64PHDRSIZE */
		hdr.shentsize = ELF64SHDRSIZE;	/* Must be ELF64SHDRSIZE */
		break;

	// 32-bit architectures
	default:
		hdr.phoff = ELF32HDRSIZE;	/* Must be be ELF32HDRSIZE: first PHdr must follow ELF header */
		hdr.shoff = ELF32HDRSIZE;	/* Will move as we add PHeaders */
		hdr.ehsize = ELF32HDRSIZE;	/* Must be ELF32HDRSIZE */
		hdr.phentsize = ELF32PHDRSIZE;	/* Must be ELF32PHDRSIZE */
		hdr.shentsize = ELF32SHDRSIZE;	/* Must be ELF32SHDRSIZE */

	}
}

void
elf64phdr(Elf64_Phdr *e)
{
	LPUT(e->type);
	LPUT(e->flags);
	VPUT(e->off);
	VPUT(e->vaddr);
	VPUT(e->paddr);
	VPUT(e->filesz);
	VPUT(e->memsz);
	VPUT(e->align);
}

void
elf64shdr(char *name, Elf64_Shdr *e)
{
	LPUT(e->name);
	LPUT(e->type);
	VPUT(e->flags);
	VPUT(e->addr);
	VPUT(e->off);
	VPUT(e->size);
	LPUT(e->link);
	LPUT(e->info);
	VPUT(e->addralign);
	VPUT(e->entsize);
}

uint32
elf64writeshdrs(void)
{
	int i;

	for (i = 0; i < hdr.shnum; i++)
		elf64shdr(sname[i], shdr[i]);
	return hdr.shnum * ELF64SHDRSIZE;
}

uint32
elf64writephdrs(void)
{
	int i;

	for (i = 0; i < hdr.phnum; i++)
		elf64phdr(phdr[i]);
	return hdr.phnum * ELF64PHDRSIZE;
}

Elf64_Phdr*
newElf64_Phdr(void)
{
	Elf64_Phdr *e;

	e = malloc(sizeof *e);
	memset(e, 0, sizeof *e);
	if (hdr.phnum >= NSECT)
		diag("too many phdrs");
	else
		phdr[hdr.phnum++] = e;
	hdr.shoff += ELF64PHDRSIZE;
	return e;
}

Elf64_Shdr*
newElf64_Shstrtab(vlong name)
{
	hdr.shstrndx = hdr.shnum;
	return newElf64_Shdr(name);
}

Elf64_Shdr*
newElf64_Shdr(vlong name)
{
	Elf64_Shdr *e;

	e = malloc(sizeof *e);
	memset(e, 0, sizeof *e);
	e->name = name;
	if (hdr.shnum >= NSECT) {
		diag("too many shdrs");
	} else {
		shdr[hdr.shnum++] = e;
	}
	return e;
}

Elf64_Ehdr*
getElf64_Ehdr(void)
{
	return &hdr;
}

uint32
elf64writehdr(void)
{
	int i;

	for (i = 0; i < EI_NIDENT; i++)
		cput(hdr.ident[i]);
	WPUT(hdr.type);
	WPUT(hdr.machine);
	LPUT(hdr.version);
	VPUT(hdr.entry);
	VPUT(hdr.phoff);
	VPUT(hdr.shoff);
	LPUT(hdr.flags);
	WPUT(hdr.ehsize);
	WPUT(hdr.phentsize);
	WPUT(hdr.phnum);
	WPUT(hdr.shentsize);
	WPUT(hdr.shnum);
	WPUT(hdr.shstrndx);
	return ELF64HDRSIZE;
}

/* Taken directly from the definition document for ELF64 */
uint32
elf64_hash(uchar *name)
{
	unsigned long h = 0, g;
	while (*name) {
		h = (h << 4) + *name++;
		if (g = h & 0xf0000000)
			h ^= g >> 24;
		h &= 0x0fffffff;
	}
	return h;
}

void
elfwritedynent(Sym *s, int tag, uint64 val)
{
	if(elf64) {
		adduint64(s, tag);
		adduint64(s, val);
	} else {
		adduint32(s, tag);
		adduint32(s, val);
	}
}

void
elfwritedynentsym(Sym *s, int tag, Sym *t)
{
	if(elf64)
		adduint64(s, tag);
	else
		adduint32(s, tag);
	addaddr(s, t);
}

void
elfwritedynentsymsize(Sym *s, int tag, Sym *t)
{
	if(elf64)
		adduint64(s, tag);
	else
		adduint32(s, tag);
	addsize(s, t);
}
