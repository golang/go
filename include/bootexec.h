// Inferno libmach/bootexec.h
// http://code.google.com/p/inferno-os/source/browse/utils/libmach/bootexec.h
//
//	Copyright © 1994-1999 Lucent Technologies Inc.
//	Power PC support Copyright © 1995-2004 C H Forsyth (forsyth@terzarima.net).
//	Portions Copyright © 1997-1999 Vita Nuova Limited.
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com).
//	Revisions Copyright © 2000-2004 Lucent Technologies Inc. and others.
//	Portions Copyright © 2009 The Go Authors. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

struct coffsect
{
	char	name[8];
	ulong	phys;
	ulong	virt;
	ulong	size;
	ulong	fptr;
	ulong	fptrreloc;
	ulong	fptrlineno;
	ulong	nrelocnlineno;
	ulong	flags;
};

/*
 * proprietary exec headers, needed to bootstrap various machines
 */
struct mipsexec
{
	short	mmagic;		/* (0x160) mips magic number */
	short	nscns;		/* (unused) number of sections */
	long	timdat;		/* (unused) time & date stamp */
	long	symptr;		/* offset to symbol table */
	long	nsyms;		/* size of symbol table */
	short	opthdr;		/* (0x38) sizeof(optional hdr) */
	short	pcszs;		/* flags */
	short	amagic;		/* see above */
	short	vstamp;		/* version stamp */
	long	tsize;		/* text size in bytes */
	long	dsize;		/* initialized data */
	long	bsize;		/* uninitialized data */
	long	mentry;		/* entry pt.				*/
	long	text_start;	/* base of text used for this file	*/
	long	data_start;	/* base of data used for this file	*/
	long	bss_start;	/* base of bss used for this file	*/
	long	gprmask;	/* general purpose register mask	*/
union{
	long	cprmask[4];	/* co-processor register masks		*/
	long	pcsize;
};
	long	gp_value;	/* the gp value used for this object    */
};

struct mips4kexec
{
	struct mipsexec	h;
	struct coffsect	itexts;
	struct coffsect idatas;
	struct coffsect ibsss;
};

struct sparcexec
{
	short	sjunk;		/* dynamic bit and version number */
	short	smagic;		/* 0407 */
	ulong	stext;
	ulong	sdata;
	ulong	sbss;
	ulong	ssyms;
	ulong	sentry;
	ulong	strsize;
	ulong	sdrsize;
};

struct nextexec
{
/* UNUSED
	struct	nexthdr{
		ulong	nmagic;
		ulong	ncputype;
		ulong	ncpusubtype;
		ulong	nfiletype;
		ulong	ncmds;
		ulong	nsizeofcmds;
		ulong	nflags;
	};

	struct nextcmd{
		ulong	cmd;
		ulong	cmdsize;
		uchar	segname[16];
		ulong	vmaddr;
		ulong	vmsize;
		ulong	fileoff;
		ulong	filesize;
		ulong	maxprot;
		ulong	initprot;
		ulong	nsects;
		ulong	flags;
	}textc;
	struct nextsect{
		char	sectname[16];
		char	segname[16];
		ulong	addr;
		ulong	size;
		ulong	offset;
		ulong	align;
		ulong	reloff;
		ulong	nreloc;
		ulong	flags;
		ulong	reserved1;
		ulong	reserved2;
	}texts;
	struct nextcmd	datac;
	struct nextsect	datas;
	struct nextsect	bsss;
	struct nextsym{
		ulong	cmd;
		ulong	cmdsize;
		ulong	symoff;
		ulong	nsyms;
		ulong	spoff;
		ulong	pcoff;
	}symc;
*/
};

struct i386exec
{
/* UNUSED
	struct	i386coff{
		ulong	isectmagic;
		ulong	itime;
		ulong	isyms;
		ulong	insyms;
		ulong	iflags;
	};
	struct	i386hdr{
		ulong	imagic;
		ulong	itextsize;
		ulong	idatasize;
		ulong	ibsssize;
		ulong	ientry;
		ulong	itextstart;
		ulong	idatastart;
	};
	struct coffsect	itexts;
	struct coffsect idatas;
	struct coffsect ibsss;
	struct coffsect icomments;
*/
};
