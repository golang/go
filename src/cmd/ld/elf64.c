// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Support for 64-bit Elf binaries

#include "../ld/elf64.h"

void
elf64phdr(int type, int flags, vlong foff,
	vlong vaddr, vlong paddr,
	vlong filesize, vlong memsize, vlong align)
{

	lputl(type);			/*  type */
	lputl(flags);			/* flags */
	vputl(foff);			/* file offset */
	vputl(vaddr);			/* vaddr */
	vputl(paddr);			/* paddr */
	vputl(filesize);		/* file size */
	vputl(memsize);		/* memory size */
	vputl(align);			/* alignment */
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

	if(name != nil)
		stroffset += strlen(name)+1;
}

int
putelf64strtab(char* name)
{
	int w;

	w = strlen(name)+1;
	strnput(name, w);
	return w;
}


int
elf64strtable(void)
{
	int size;

	size = 0;
	size += putelf64strtab("");
	size += putelf64strtab(".text");
	size += putelf64strtab(".data");
	size += putelf64strtab(".bss");
	size += putelf64strtab(".shstrtab");
	if (!debug['s']) {
		size += putelf64strtab(".gosymtab");
		size += putelf64strtab(".gopclntab");
	}
	return size;
}


uint32
elf64headr(void)
{
	uint32 a;

	a = 64;		/* a.out header */

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

Elf64SHdr*
newElf64SHdr()
{
	Elf64SHdr *e;

	e = malloc(sizeof *e);
	memset(e, 0, sizeof *e);
	e->name = stroffset;
	return e;
}

