// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 *	Definitions needed for  accessing MACH object headers.
 */

typedef struct {
	ulong	magic;		/* mach magic number identifier */
	ulong	cputype;	/* cpu specifier */
	ulong	cpusubtype;	/* machine specifier */
	ulong	filetype;	/* type of file */
	ulong	ncmds;		/* number of load commands */
	ulong	sizeofcmds;	/* the size of all the load commands */
	ulong	flags;		/* flags */
	ulong	reserved;	/* reserved */
} Machhdr;

typedef struct {
	ulong	type;	/* type of load command */
	ulong	size;	/* total size in bytes */
} MachCmd;

typedef struct  {
	MachCmd	cmd;
	char		segname[16];	/* segment name */
	uvlong	vmaddr;		/* memory address of this segment */
	uvlong	vmsize;		/* memory size of this segment */
	uvlong	fileoff;	/* file offset of this segment */
	uvlong	filesize;	/* amount to map from the file */
	ulong	maxprot;	/* maximum VM protection */
	ulong	initprot;	/* initial VM protection */
	ulong	nsects;		/* number of sections in segment */
	ulong	flags;		/* flags */
} MachSeg64; /* for 64-bit architectures */

typedef struct  {
	MachCmd	cmd;
	ulong	fileoff;	/* file offset of this segment */
	ulong	filesize;	/* amount to map from the file */
} MachSymSeg;

typedef struct  {
	char		sectname[16];	/* name of this section */
	char		segname[16];	/* segment this section goes in */
	uvlong	addr;		/* memory address of this section */
	uvlong	size;		/* size in bytes of this section */
	ulong	offset;		/* file offset of this section */
	ulong	align;		/* section alignment (power of 2) */
	ulong	reloff;		/* file offset of relocation entries */
	ulong	nreloc;		/* number of relocation entries */
	ulong	flags;		/* flags (section type and attributes)*/
	ulong	reserved1;	/* reserved (for offset or index) */
	ulong	reserved2;	/* reserved (for count or sizeof) */
	ulong	reserved3;	/* reserved */
} MachSect64; /* for 64-bit architectures */

enum {
	MACH_CPU_TYPE_X86_64 = (1<<24)|7,
	MACH_CPU_SUBTYPE_X86 = 3,
	MACH_EXECUTABLE_TYPE = 2,
	MACH_SEGMENT_64 = 0x19,	/* 64-bit mapped segment */
	MACH_SYMSEG = 3,	/* obsolete gdb symtab, reused by go */
	MACH_UNIXTHREAD = 0x5,	/* thread (for stack) */
};


#define	MACH_MAG		((0xcf<<24) | (0xfa<<16) | (0xed<<8) | 0xfe)
