// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Support for 64-bit Elf binaries

#include "../ld/elf64.h"

#define	NSECT	16
static	int	numstr;
static	int	stroffset;
static	Elf64Hdr	hdr;
static	Elf64PHdr	*phdr[NSECT];
static	Elf64SHdr	*shdr[NSECT];
static	char	*sname[NSECT];
static	char	*str[NSECT];

/*
 Initialize the global variable that describes the ELF header. It will be updated as
 we write section and prog headers.
 */
void
elf64init(void)
{
	hdr.phoff = ELF64HDRSIZE;	/* Must be be ELF64HDRSIZE: first PHdr must follow ELF header */
	hdr.shoff = ELF64HDRSIZE;	/* Will move as we add PHeaders */
	hdr.ehsize = ELF64HDRSIZE;	/* Must be ELF64HDRSIZE */
	hdr.phentsize = ELF64PHDRSIZE;	/* Must be ELF64PHDRSIZE */
	hdr.shentsize = ELF64SHDRSIZE;	/* Must be ELF64SHDRSIZE */
}

void
elf64phdr(Elf64PHdr *e)
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
elf64shdr(char *name, Elf64SHdr *e)
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

int
putelf64strtab(char* name)
{
	int w;

	w = strlen(name)+1;
	strnput(name, w);
	return w;
}

void
elf64writestrtable(void)
{
	int i;
	uint32 size;

	size = 0;
	for (i = 0; i < numstr; i++)
		size += putelf64strtab(str[i]);
	if (size > STRTABSIZE)
		diag("elf64 string table overflow");
}

void
e64addstr(char *name)
{
	if (numstr >= NSECT) {
		diag("too many elf strings");
		return;
	}
	str[numstr++] = strdup(name);
	stroffset += strlen(name)+1;
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

Elf64PHdr*
newElf64PHdr(void)
{
	Elf64PHdr *e;

	e = malloc(sizeof *e);
	memset(e, 0, sizeof *e);
	if (hdr.phnum >= NSECT)
		diag("too many phdrs");
	else
		phdr[hdr.phnum++] = e;
	hdr.shoff += ELF64PHDRSIZE;
	return e;
}

Elf64SHdr*
newElf64SHdr(char *name)
{
	Elf64SHdr *e;

	if (strcmp(name, ".shstrtab") == 0)
		hdr.shstrndx = hdr.shnum;
	e = malloc(sizeof *e);
	memset(e, 0, sizeof *e);
	e->name = stroffset;
	if (hdr.shnum >= NSECT) {
		diag("too many shdrs");
	} else {
		e64addstr(name);
		shdr[hdr.shnum++] = e;
	}
	return e;
}

Elf64Hdr*
getElf64Hdr(void)
{
	return &hdr;
}

uint32
elf64writehdr()
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
elf64writedynent(int tag, uint64 val)
{
	VPUT(tag);
	VPUT(val);
}

/* Where to write the next piece of data attached to an SHeader */
uint64	elfaddr = ELF64FULLHDRSIZE;

/* Mark a start location in the SHeader data */
uint64
startelf(void)
{
	seek(cout, elfaddr, 0);
	return elfaddr;
}

/* Mark the end of a location in the SHeader data */
uint64
endelf(void)
{
	uint64 p;

	cflush();
	p = seek(cout, 0, 1);
	if (p < elfaddr) {
		diag("endelf before elfaddr");
	}
	if ((p & 7) != 0) {
		p = (p + 7) & ~7LL;
		seek(cout, p, 0);
	}
	elfaddr = p;
	if (p > ELF64RESERVE) {
		diag("endelf overflows reserve %lld\n", p);
	}
	return elfaddr;
}
