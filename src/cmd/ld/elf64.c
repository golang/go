// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Support for 64-bit Elf binaries

#include "../ld/elf64.h"

#define	NSECT	16
static	int	nume64str;
static	Elf64Hdr	hdr;
static	Elf64PHdr	*phdr[NSECT];
static	Elf64SHdr	*shdr[NSECT];
static	char	*sname[NSECT];
static	char	*str[NSECT];

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
	int size;

	size = 0;
	for (i = 0; i < nume64str; i++)
		size += putelf64strtab(str[i]);
	if (size > STRTABSIZE)
		diag("elf64 string table overflow");
}

void
e64addstr(char *name)
{
	if (nume64str >= NSECT) {
		diag("too many elf strings");
		return;
	}
	str[nume64str++] = strdup(name);
	stroffset += strlen(name)+1;
}

uint32
elf64headr(void)
{
	uint32 a;

	a = 64;		/* a.out header */

	/* TODO: calculate these byte counts properly */
	a += 56;	/* page zero seg */
	a += 56;	/* text seg */
	a += 56;	/* stack seg */

	a += 64;	/* nil sect */
	a += 64;	/* .text sect */
	a += 64;	/* .data seg */
	a += 64;	/* .bss sect */
	a += 64;	/* .shstrtab sect - strings for headers */
	if (!debug['s']) {
		a += 56;	/* symdat seg */
		a += 64;	/* .gosymtab sect */
		a += 64;	/* .gopclntab sect */
	}

	return a;
}

void
elf64writeshdrs(void)
{
	int i;

	for (i = 0; i < hdr.shnum; i++)
		elf64shdr(sname[i], shdr[i]);
}

void
elf64writephdrs(void)
{
	int i;

	for (i = 0; i < hdr.phnum; i++)
		elf64phdr(phdr[i]);
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

void
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
}
