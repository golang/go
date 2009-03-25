// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 *	Definitions needed for  accessing MACH object headers.
 */

typedef struct {
	uint32	magic;		/* mach magic number identifier */
	uint32	cputype;	/* cpu specifier */
	uint32	cpusubtype;	/* machine specifier */
	uint32	filetype;	/* type of file */
	uint32	ncmds;		/* number of load commands */
	uint32	sizeofcmds;	/* the size of all the load commands */
	uint32	flags;		/* flags */
	uint32	reserved;	/* reserved */
} Machhdr;

typedef struct {
	uint32	type;	/* type of load command */
	uint32	size;	/* total size in bytes */
} MachCmd;

typedef struct  {
	MachCmd	cmd;
	char		segname[16];	/* segment name */
	uint32	vmaddr;		/* memory address of this segment */
	uint32	vmsize;		/* memory size of this segment */
	uint32	fileoff;	/* file offset of this segment */
	uint32	filesize;	/* amount to map from the file */
	uint32	maxprot;	/* maximum VM protection */
	uint32	initprot;	/* initial VM protection */
	uint32	nsects;		/* number of sections in segment */
	uint32	flags;		/* flags */
} MachSeg32; /* for 32-bit architectures */

typedef struct  {
	MachCmd	cmd;
	char		segname[16];	/* segment name */
	uvlong	vmaddr;		/* memory address of this segment */
	uvlong	vmsize;		/* memory size of this segment */
	uvlong	fileoff;	/* file offset of this segment */
	uvlong	filesize;	/* amount to map from the file */
	uint32	maxprot;	/* maximum VM protection */
	uint32	initprot;	/* initial VM protection */
	uint32	nsects;		/* number of sections in segment */
	uint32	flags;		/* flags */
} MachSeg64; /* for 64-bit architectures */

typedef struct  {
	MachCmd	cmd;
	uint32	fileoff;	/* file offset of this segment */
	uint32	filesize;	/* amount to map from the file */
} MachSymSeg;

typedef struct  {
	char		sectname[16];	/* name of this section */
	char		segname[16];	/* segment this section goes in */
	uint32	addr;		/* memory address of this section */
	uint32	size;		/* size in bytes of this section */
	uint32	offset;		/* file offset of this section */
	uint32	align;		/* section alignment (power of 2) */
	uint32	reloff;		/* file offset of relocation entries */
	uint32	nreloc;		/* number of relocation entries */
	uint32	flags;		/* flags (section type and attributes)*/
	uint32	reserved1;	/* reserved (for offset or index) */
	uint32	reserved2;	/* reserved (for count or sizeof) */
} MachSect32; /* for 32-bit architectures */

typedef struct  {
	char		sectname[16];	/* name of this section */
	char		segname[16];	/* segment this section goes in */
	uvlong	addr;		/* memory address of this section */
	uvlong	size;		/* size in bytes of this section */
	uint32	offset;		/* file offset of this section */
	uint32	align;		/* section alignment (power of 2) */
	uint32	reloff;		/* file offset of relocation entries */
	uint32	nreloc;		/* number of relocation entries */
	uint32	flags;		/* flags (section type and attributes)*/
	uint32	reserved1;	/* reserved (for offset or index) */
	uint32	reserved2;	/* reserved (for count or sizeof) */
	uint32	reserved3;	/* reserved */
} MachSect64; /* for 64-bit architectures */

enum {
	MACH_CPU_TYPE_X86_64 = (1<<24)|7,
	MACH_CPU_TYPE_X86 = 7,
	MACH_CPU_SUBTYPE_X86 = 3,
	MACH_EXECUTABLE_TYPE = 2,
	MACH_SEGMENT_32 = 1,	/* 32-bit mapped segment */
	MACH_SEGMENT_64 = 0x19,	/* 64-bit mapped segment */
	MACH_SYMSEG = 3,	/* obsolete gdb symtab, reused by go */
	MACH_UNIXTHREAD = 0x5,	/* thread (for stack) */
};


#define	MACH64_MAG		((0xcf<<24) | (0xfa<<16) | (0xed<<8) | 0xfe)
#define	MACH32_MAG		((0xce<<24) | (0xfa<<16) | (0xed<<8) | 0xfe)
