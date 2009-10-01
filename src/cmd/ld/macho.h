// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

typedef struct MachoHdr MachoHdr;
struct MachoHdr {
	uint32	cpu;
	uint32	subcpu;
};

typedef struct MachoSect MachoSect;
struct MachoSect {
	char*	name;
	uint64	addr;
	uint64	size;
	uint32	off;
	uint32	align;
	uint32	reloc;
	uint32	nreloc;
	uint32	flag;
};

typedef struct MachoSeg MachoSeg;
struct MachoSeg {
	char*	name;
	uint64	vsize;
	uint64	vaddr;
	uint64	fileoffset;
	uint64	filesize;
	uint32	prot1;
	uint32	prot2;
	uint32	nsect;
	uint32	msect;
	MachoSect	*sect;
	uint32	flag;
};

typedef struct MachoLoad MachoLoad;
struct MachoLoad {
	uint32	type;
	uint32	ndata;
	uint32	*data;
};

typedef struct MachoDebug MachoDebug;
struct MachoDebug {
	uint32	fileoffset;
	uint32	filesize;
};

MachoHdr*	getMachoHdr();
MachoSeg*	newMachoSeg(char*, int);
MachoSect*	newMachoSect(MachoSeg*, char*);
MachoLoad*	newMachoLoad(uint32, uint32);
MachoDebug*	newMachoDebug(void);
int	machowrite(void);
void	machoinit(void);

/*
 * Total amount of space to reserve at the start of the file
 * for Header, PHeaders, and SHeaders.
 * May waste some.
 */
#define	MACHORESERVE	4096

enum {
	MACHO_CPU_AMD64 = (1<<24)|7,
	MACHO_CPU_386 = 7,
	MACHO_SUBCPU_X86 = 3,
};
