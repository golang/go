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

typedef struct Elf64Hdr Elf64Hdr;
typedef struct Elf64SHdr Elf64SHdr;

struct Elf64Hdr
{
	uchar ident[16];	/* ELF identification */
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

struct Elf64SHdr
{
	Elf64_Word	name;	/* Section name */
	Elf64_Word	type;	/* Section type */
	Elf64_Xword	flags;	/* Section attributes */
	Elf64_Addr	addr;	/* Virtual address in memory */
	Elf64_Off	off; /* Offset in file */
	Elf64_Xword	size;	/* Size of section */
	Elf64_Word	link;	/* Link to other section */
	Elf64_Word	info;	/* Miscellaneous information */
	Elf64_Xword	addralign;	/* Address alignment boundary */
	Elf64_Xword	entsize;	/* Size of entries, if section has table */
};

Elf64SHdr *newElf64SHdr();
uint32	elf64headr(void);
void	elf64phdr(int type, int flags, vlong foff,
	vlong vaddr, vlong paddr,
	vlong filesize, vlong memsize, vlong align);
void	elf64shdr(char*, Elf64SHdr*);
int	elf64strtable(void);
