// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

/*
Input to godefs.  See also mkerrors.sh and mkall.sh
*/

typedef unsigned short ushort;
typedef unsigned char uchar;
typedef unsigned long ulong;
typedef unsigned int uint;
typedef long long vlong;
typedef unsigned long long uvlong;

typedef int $_C_int;

enum {
	OREAD	= 0,	// open for read
	OWRITE	= 1,	// write
	ORDWR	= 2,	// read and write
	OEXEC	= 3,	// execute, == read but check execute permission
	OTRUNC	= 16,	// or'ed in (except for exec), truncate file first
	OCEXEC	= 32,	// or'ed in, close on exec
	ORCLOSE	= 64,		// or'ed in, remove on close
	OEXCL	= 0x1000,	// or'ed in, exclusive use (create only)

	$O_RDONLY	= OREAD,
	$O_WRONLY	= OWRITE,
	$O_RDWR		= ORDWR,
	$O_TRUNC	= OTRUNC,
	$O_CLOEXEC	= OCEXEC,
	$O_EXCL		= OEXCL,

	$STATMAX	= 65535U,
	$ERRMAX		= 128,

	$MORDER		= 0x0003,	// mask for bits defining order of mounting
	$MREPL		= 0x0000,	// mount replaces object
	$MBEFORE	= 0x0001,	// mount goes before others in union directory
	$MAFTER		= 0x0002,	// mount goes after others in union directory
	$MCREATE	= 0x0004,	// permit creation in mounted directory
	$MCACHE		= 0x0010,	// cache some data
	$MMASK		= 0x0017,	// all bits on

	$RFNAMEG	= (1<<0),
	$RFENVG		= (1<<1),
	$RFFDG		= (1<<2),
	$RFNOTEG	= (1<<3),
	$RFPROC		= (1<<4),
	$RFMEM		= (1<<5),
	$RFNOWAIT	= (1<<6),
	$RFCNAMEG	= (1<<10),
	$RFCENVG	= (1<<11),
	$RFCFDG		= (1<<12),
	$RFREND		= (1<<13),
	$RFNOMNT	= (1<<14),

	// bits in Qid.type
	$QTDIR		= 0x80,		// type bit for directories
	$QTAPPEND	= 0x40,		// type bit for append only files
	$QTEXCL		= 0x20,		// type bit for exclusive use files
	$QTMOUNT	= 0x10,		// type bit for mounted channel
	$QTAUTH		= 0x08,		// type bit for authentication file
	$QTTMP		= 0x04,		// type bit for not-backed-up file
	$QTFILE		= 0x00,		// plain file


	// bits in Dir.mode
	$DMDIR		= 0x80000000,	// mode bit for directories
	$DMAPPEND	= 0x40000000,	// mode bit for append only files
	$DMEXCL		= 0x20000000,	// mode bit for exclusive use files
	$DMMOUNT	= 0x10000000,	// mode bit for mounted channel
	$DMAUTH		= 0x08000000,	// mode bit for authentication file
	$DMTMP		= 0x04000000,	// mode bit for non-backed-up files
	$DMREAD		= 0x4,		// mode bit for read permission
	$DMWRITE	= 0x2,		// mode bit for write permission
	$DMEXEC		= 0x1,		// mode bit for execute permission

	BIT8SZ	= 1,
	BIT16SZ	= 2,
	BIT32SZ	= 4,
	BIT64SZ	= 8,
	QIDSZ = BIT8SZ+BIT32SZ+BIT64SZ,

	// STATFIXLEN includes leading 16-bit count
	// The count, however, excludes itself; total size is BIT16SZ+count
	$STATFIXLEN = BIT16SZ+QIDSZ+5*BIT16SZ+4*BIT32SZ+1*BIT64SZ,	// amount of fixed length data in a stat buffer
};


struct Prof			// Per process profiling
{
	struct Plink	*pp;	// known to be 0(ptr)
	struct Plink	*next;	// known to be 4(ptr)
	struct Plink	*last;
	struct Plink	*first;
	ulong		pid;
	ulong		what;
};

struct Tos {
	struct Prof	prof;
	uvlong		cyclefreq;	// cycle clock frequency if there is one, 0 otherwise
	vlong		kcycles;	// cycles spent in kernel
	vlong		pcycles;	// cycles spent in process (kernel + user)
	ulong		pid;		// might as well put the pid here
	ulong		clock;
	// top of stack is here
};

typedef struct Prof $Prof;
typedef struct Tos $Tos;
