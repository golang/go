// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Support for 64-bit Elf binaries

#include "../ld/elf64.h"

#define	NSECT	16
int	nume64phdr;
int	nume64shdr;
int	nume64str;
static	Elf64PHdr	*phdr[NSECT];
static	Elf64SHdr	*shdr[NSECT];
static	char	*sname[NSECT];
static	char	*str[NSECT];

void
elf64phdr(Elf64PHdr *e)
{
	lputl(e->type);
	lputl(e->flags);
	vputl(e->off);
	vputl(e->vaddr);
	vputl(e->paddr);
	vputl(e->filesz);
	vputl(e->memsz);
	vputl(e->align);
}

void
elf64shdr(char *name, Elf64SHdr *e)
{
	lputl(e->name);
	lputl(e->type);
	vputl(e->flags);
	vputl(e->addr);
	vputl(e->off);
	vputl(e->size);
	lputl(e->link);
	lputl(e->info);
	vputl(e->addralign);
	vputl(e->entsize);
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

	for (i = 0; i < nume64shdr; i++)
		elf64shdr(sname[i], shdr[i]);
}

void
elf64writephdrs(void)
{
	int i;

	for (i = 0; i < nume64phdr; i++)
		elf64phdr(phdr[i]);
}

Elf64PHdr*
newElf64PHdr(void)
{
	Elf64PHdr *e;

	e = malloc(sizeof *e);
	memset(e, 0, sizeof *e);
	if (nume64phdr >= NSECT)
		diag("too many phdrs");
	else
		phdr[nume64phdr++] = e;
	return e;
}

Elf64SHdr*
newElf64SHdr(char *name)
{
	Elf64SHdr *e;

	e = malloc(sizeof *e);
	memset(e, 0, sizeof *e);
	e->name = stroffset;
	if (nume64shdr >= NSECT) {
		diag("too many shdrs");
	} else {
		e64addstr(name);
		shdr[nume64shdr++] = e;
	}
	return e;
}
