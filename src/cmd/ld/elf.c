// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	"l.h"
#include	"lib.h"
#include	"../ld/elf.h"

/*
 * We use the 64-bit data structures on both 32- and 64-bit machines
 * in order to write the code just once.  The 64-bit data structure is
 * written in the 32-bit format on the 32-bit machines.
 */
#define	NSECT	32

static	int	elf64;
static	ElfEhdr	hdr;
static	ElfPhdr	*phdr[NSECT];
static	ElfShdr	*shdr[NSECT];

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
elf64phdr(ElfPhdr *e)
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
elf32phdr(ElfPhdr *e)
{
	LPUT(e->type);
	LPUT(e->off);
	LPUT(e->vaddr);
	LPUT(e->paddr);
	LPUT(e->filesz);
	LPUT(e->memsz);
	LPUT(e->flags);
	LPUT(e->align);
}

void
elf64shdr(ElfShdr *e)
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

void
elf32shdr(ElfShdr *e)
{
	LPUT(e->name);
	LPUT(e->type);
	LPUT(e->flags);
	LPUT(e->addr);
	LPUT(e->off);
	LPUT(e->size);
	LPUT(e->link);
	LPUT(e->info);
	LPUT(e->addralign);
	LPUT(e->entsize);
}

uint32
elfwriteshdrs(void)
{
	int i;

	if (elf64) {
		for (i = 0; i < hdr.shnum; i++)
			elf64shdr(shdr[i]);
		return hdr.shnum * ELF64SHDRSIZE;
	}
	for (i = 0; i < hdr.shnum; i++)
		elf32shdr(shdr[i]);
	return hdr.shnum * ELF32SHDRSIZE;
}

uint32
elfwritephdrs(void)
{
	int i;

	if (elf64) {
		for (i = 0; i < hdr.phnum; i++)
			elf64phdr(phdr[i]);
		return hdr.phnum * ELF64PHDRSIZE;
	}
	for (i = 0; i < hdr.phnum; i++)
		elf32phdr(phdr[i]);
	return hdr.phnum * ELF32PHDRSIZE;
}

ElfPhdr*
newElfPhdr(void)
{
	ElfPhdr *e;

	e = malloc(sizeof *e);
	memset(e, 0, sizeof *e);
	if (hdr.phnum >= NSECT)
		diag("too many phdrs");
	else
		phdr[hdr.phnum++] = e;
	if (elf64)
		hdr.shoff += ELF64PHDRSIZE;
	else
		hdr.shoff += ELF32PHDRSIZE;
	return e;
}

ElfShdr*
newElfShstrtab(vlong name)
{
	hdr.shstrndx = hdr.shnum;
	return newElfShdr(name);
}

ElfShdr*
newElfShdr(vlong name)
{
	ElfShdr *e;

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

ElfEhdr*
getElfEhdr(void)
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

uint32
elf32writehdr(void)
{
	int i;

	for (i = 0; i < EI_NIDENT; i++)
		cput(hdr.ident[i]);
	WPUT(hdr.type);
	WPUT(hdr.machine);
	LPUT(hdr.version);
	LPUT(hdr.entry);
	LPUT(hdr.phoff);
	LPUT(hdr.shoff);
	LPUT(hdr.flags);
	WPUT(hdr.ehsize);
	WPUT(hdr.phentsize);
	WPUT(hdr.phnum);
	WPUT(hdr.shentsize);
	WPUT(hdr.shnum);
	WPUT(hdr.shstrndx);
	return ELF32HDRSIZE;
}

uint32
elfwritehdr(void)
{
	if(elf64)
		return elf64writehdr();
	return elf32writehdr();
}

/* Taken directly from the definition document for ELF64 */
uint32
elfhash(uchar *name)
{
	uint32 h = 0, g;
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
