// Inferno libmach/executable.c
// http://code.google.com/p/inferno-os/source/browse/utils/libmach/executable.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.
//	Power PC support Copyright © 1995-2004 C H Forsyth (forsyth@terzarima.net).
//	Portions Copyright © 1997-1999 Vita Nuova Limited.
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com).
//	Revisions Copyright © 2000-2004 Lucent Technologies Inc. and others.
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
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

#include	<u.h>
#include	<libc.h>
#include	<bio.h>
#include	<bootexec.h>
#include	<mach_amd64.h>
#include	"elf.h"

/*
 *	All a.out header types.  The dummy entry allows canonical
 *	processing of the union as a sequence of longs
 */

typedef struct {
	union{
		/*struct { */
			Exec exechdr;		/* a.out.h */
		/*	uvlong hdr[1];*/
		/*};*/
		Ehdr32 elfhdr32;			/* elf.h */
		Ehdr64 elfhdr64;			/* elf.h */
		struct mipsexec mips;	/* bootexec.h */
		struct mips4kexec mipsk4;	/* bootexec.h */
		struct sparcexec sparc;	/* bootexec.h */
		struct nextexec next;	/* bootexec.h */
	} e;
	long dummy;			/* padding to ensure extra long */
} ExecHdr;

static	int	nextboot(int, Fhdr*, ExecHdr*);
static	int	sparcboot(int, Fhdr*, ExecHdr*);
static	int	mipsboot(int, Fhdr*, ExecHdr*);
static	int	mips4kboot(int, Fhdr*, ExecHdr*);
static	int	common(int, Fhdr*, ExecHdr*);
static	int	commonllp64(int, Fhdr*, ExecHdr*);
static	int	adotout(int, Fhdr*, ExecHdr*);
static	int	elfdotout(int, Fhdr*, ExecHdr*);
static	int	armdotout(int, Fhdr*, ExecHdr*);
static	void	setsym(Fhdr*, long, long, long, vlong);
static	void	setdata(Fhdr*, uvlong, long, vlong, long);
static	void	settext(Fhdr*, uvlong, uvlong, long, vlong);
static	void	hswal(void*, int, ulong(*)(ulong));
static	uvlong	_round(uvlong, ulong);

/*
 *	definition of per-executable file type structures
 */

typedef struct Exectable{
	long	magic;			/* big-endian magic number of file */
	char	*name;			/* executable identifier */
	char	*dlmname;		/* dynamically loadable module identifier */
	uchar	type;			/* Internal code */
	uchar	_magic;			/* _MAGIC() magic */
	Mach	*mach;			/* Per-machine data */
	long	hsize;			/* header size */
	ulong	(*swal)(ulong);		/* beswal or leswal */
	int	(*hparse)(int, Fhdr*, ExecHdr*);
} ExecTable;

extern	Mach	mmips;
extern	Mach	mmips2le;
extern	Mach	mmips2be;
extern	Mach	msparc;
extern	Mach	msparc64;
extern	Mach	m68020;
extern	Mach	mi386;
extern	Mach	mamd64;
extern	Mach	marm;
extern	Mach	mpower;
extern	Mach	mpower64;
extern	Mach	malpha;

/* BUG: FIX THESE WHEN NEEDED */
Mach	mmips;
Mach	mmips2le;
Mach	mmips2be;
Mach	msparc;
Mach	msparc64;
Mach	m68020;
Mach	marm;
Mach	mpower;
Mach	mpower64;
Mach	malpha;

ExecTable exectab[] =
{
	{ V_MAGIC,			/* Mips v.out */
		"mips plan 9 executable BE",
		"mips plan 9 dlm BE",
		FMIPS,
		1,
		&mmips,
		sizeof(Exec),
		beswal,
		adotout },
	{ P_MAGIC,			/* Mips 0.out (r3k le) */
		"mips plan 9 executable LE",
		"mips plan 9 dlm LE",
		FMIPSLE,
		1,
		&mmips,
		sizeof(Exec),
		beswal,
		adotout },
	{ M_MAGIC,			/* Mips 4.out */
		"mips 4k plan 9 executable BE",
		"mips 4k plan 9 dlm BE",
		FMIPS2BE,
		1,
		&mmips2be,
		sizeof(Exec),
		beswal,
		adotout },
	{ N_MAGIC,			/* Mips 0.out */
		"mips 4k plan 9 executable LE",
		"mips 4k plan 9 dlm LE",
		FMIPS2LE,
		1,
		&mmips2le,
		sizeof(Exec),
		beswal,
		adotout },
	{ 0x160<<16,			/* Mips boot image */
		"mips plan 9 boot image",
		nil,
		FMIPSB,
		0,
		&mmips,
		sizeof(struct mipsexec),
		beswal,
		mipsboot },
	{ (0x160<<16)|3,		/* Mips boot image */
		"mips 4k plan 9 boot image",
		nil,
		FMIPSB,
		0,
		&mmips2be,
		sizeof(struct mips4kexec),
		beswal,
		mips4kboot },
	{ K_MAGIC,			/* Sparc k.out */
		"sparc plan 9 executable",
		"sparc plan 9 dlm",
		FSPARC,
		1,
		&msparc,
		sizeof(Exec),
		beswal,
		adotout },
	{ 0x01030107, 			/* Sparc boot image */
		"sparc plan 9 boot image",
		nil,
		FSPARCB,
		0,
		&msparc,
		sizeof(struct sparcexec),
		beswal,
		sparcboot },
	{ U_MAGIC,			/* Sparc64 u.out */
		"sparc64 plan 9 executable",
		"sparc64 plan 9 dlm",
		FSPARC64,
		1,
		&msparc64,
		sizeof(Exec),
		beswal,
		adotout },
	{ A_MAGIC,			/* 68020 2.out & boot image */
		"68020 plan 9 executable",
		"68020 plan 9 dlm",
		F68020,
		1,
		&m68020,
		sizeof(Exec),
		beswal,
		common },
	{ 0xFEEDFACE,			/* Next boot image */
		"next plan 9 boot image",
		nil,
		FNEXTB,
		0,
		&m68020,
		sizeof(struct nextexec),
		beswal,
		nextboot },
	{ I_MAGIC,			/* I386 8.out & boot image */
		"386 plan 9 executable",
		"386 plan 9 dlm",
		FI386,
		1,
		&mi386,
		sizeof(Exec),
		beswal,
		common },
	{ S_MAGIC,			/* amd64 6.out & boot image */
		"amd64 plan 9 executable",
		"amd64 plan 9 dlm",
		FAMD64,
		1,
		&mamd64,
		sizeof(Exec)+8,
		nil,
		commonllp64 },
	{ Q_MAGIC,			/* PowerPC q.out & boot image */
		"power plan 9 executable",
		"power plan 9 dlm",
		FPOWER,
		1,
		&mpower,
		sizeof(Exec),
		beswal,
		common },
	{ T_MAGIC,			/* power64 9.out & boot image */
		"power64 plan 9 executable",
		"power64 plan 9 dlm",
		FPOWER64,
		1,
		&mpower64,
		sizeof(Exec)+8,
		nil,
		commonllp64 },
	{ ELF_MAG,			/* any elf32 or elf64 */
		"elf executable",
		nil,
		FNONE,
		0,
		&mi386,
		sizeof(Ehdr64),
		nil,
		elfdotout },
	{ E_MAGIC,			/* Arm 5.out and boot image */
		"arm plan 9 executable",
		"arm plan 9 dlm",
		FARM,
		1,
		&marm,
		sizeof(Exec),
		beswal,
		common },
	{ (143<<16)|0413,		/* (Free|Net)BSD Arm */
		"arm *bsd executable",
		nil,
		FARM,
		0,
		&marm,
		sizeof(Exec),
		leswal,
		armdotout },
	{ L_MAGIC,			/* alpha 7.out */
		"alpha plan 9 executable",
		"alpha plan 9 dlm",
		FALPHA,
		1,
		&malpha,
		sizeof(Exec),
		beswal,
		common },
	{ 0x0700e0c3,			/* alpha boot image */
		"alpha plan 9 boot image",
		nil,
		FALPHA,
		0,
		&malpha,
		sizeof(Exec),
		beswal,
		common },
	{ 0 },
};

Mach	*mach = &mi386;			/* Global current machine table */

static ExecTable*
couldbe4k(ExecTable *mp)
{
	Dir *d;
	ExecTable *f;

	if((d=dirstat("/proc/1/regs")) == nil)
		return mp;
	if(d->length < 32*8){		/* R3000 */
		free(d);
		return mp;
	}
	free(d);
	for (f = exectab; f->magic; f++)
		if(f->magic == M_MAGIC) {
			f->name = "mips plan 9 executable on mips2 kernel";
			return f;
		}
	return mp;
}

int
crackhdr(int fd, Fhdr *fp)
{
	ExecTable *mp;
	ExecHdr d;
	int nb, ret;
	ulong magic;

	fp->type = FNONE;
	nb = read(fd, (char *)&d.e, sizeof(d.e));
	if (nb <= 0)
		return 0;

	ret = 0;
	magic = beswal(d.e.exechdr.magic);		/* big-endian */
	for (mp = exectab; mp->magic; mp++) {
		if (nb < mp->hsize)
			continue;

		/*
		 * The magic number has morphed into something
		 * with fields (the straw was DYN_MAGIC) so now
		 * a flag is needed in Fhdr to distinguish _MAGIC()
		 * magic numbers from foreign magic numbers.
		 *
		 * This code is creaking a bit and if it has to
		 * be modified/extended much more it's probably
		 * time to step back and redo it all.
		 */
		if(mp->_magic){
			if(mp->magic != (magic & ~DYN_MAGIC))
				continue;

			if(mp->magic == V_MAGIC)
				mp = couldbe4k(mp);

			if ((magic & DYN_MAGIC) && mp->dlmname != nil)
				fp->name = mp->dlmname;
			else
				fp->name = mp->name;
		}
		else{
			if(mp->magic != magic)
				continue;
			fp->name = mp->name;
		}
		fp->type = mp->type;
		fp->hdrsz = mp->hsize;		/* will be zero on bootables */
		fp->_magic = mp->_magic;
		fp->magic = magic;

		mach = mp->mach;
		if(mp->swal != nil)
			hswal(&d, sizeof(d.e)/sizeof(ulong), mp->swal);
		ret = mp->hparse(fd, fp, &d);
		seek(fd, mp->hsize, 0);		/* seek to end of header */
		break;
	}
	if(mp->magic == 0)
		werrstr("unknown header type");
	return ret;
}

/*
 * Convert header to canonical form
 */
static void
hswal(void *v, int n, ulong (*swap)(ulong))
{
	ulong *ulp;

	for(ulp = v; n--; ulp++)
		*ulp = (*swap)(*ulp);
}

/*
 *	Crack a normal a.out-type header
 */
static int
adotout(int fd, Fhdr *fp, ExecHdr *hp)
{
	long pgsize;

	USED(fd);
	pgsize = mach->pgsize;
	settext(fp, hp->e.exechdr.entry, pgsize+sizeof(Exec),
			hp->e.exechdr.text, sizeof(Exec));
	setdata(fp, _round(pgsize+fp->txtsz+sizeof(Exec), pgsize),
		hp->e.exechdr.data, fp->txtsz+sizeof(Exec), hp->e.exechdr.bss);
	setsym(fp, hp->e.exechdr.syms, hp->e.exechdr.spsz, hp->e.exechdr.pcsz, fp->datoff+fp->datsz);
	return 1;
}

static void
commonboot(Fhdr *fp)
{
	if (!(fp->entry & mach->ktmask))
		return;

	switch(fp->type) {				/* boot image */
	case F68020:
		fp->type = F68020B;
		fp->name = "68020 plan 9 boot image";
		break;
	case FI386:
		fp->type = FI386B;
		fp->txtaddr = (u32int)fp->entry;
		fp->name = "386 plan 9 boot image";
		fp->dataddr = _round(fp->txtaddr+fp->txtsz, mach->pgsize);
		break;
	case FARM:
		fp->type = FARMB;
		fp->txtaddr = (u32int)fp->entry;
		fp->name = "ARM plan 9 boot image";
		fp->dataddr = _round(fp->txtaddr+fp->txtsz, mach->pgsize);
		return;
	case FALPHA:
		fp->type = FALPHAB;
		fp->txtaddr = (u32int)fp->entry;
		fp->name = "alpha plan 9 boot image";
		fp->dataddr = fp->txtaddr+fp->txtsz;
		break;
	case FPOWER:
		fp->type = FPOWERB;
		fp->txtaddr = (u32int)fp->entry;
		fp->name = "power plan 9 boot image";
		fp->dataddr = fp->txtaddr+fp->txtsz;
		break;
	case FAMD64:
		fp->type = FAMD64B;
		fp->txtaddr = fp->entry;
		fp->name = "amd64 plan 9 boot image";
		fp->dataddr = _round(fp->txtaddr+fp->txtsz, mach->pgsize);
		break;
	default:
		return;
	}
	fp->hdrsz = 0;			/* header stripped */
}

/*
 *	_MAGIC() style headers and
 *	alpha plan9-style bootable images for axp "headerless" boot
 *
 */
static int
common(int fd, Fhdr *fp, ExecHdr *hp)
{
	adotout(fd, fp, hp);
	if(hp->e.exechdr.magic & DYN_MAGIC) {
		fp->txtaddr = 0;
		fp->dataddr = fp->txtsz;
		return 1;
	}
	commonboot(fp);
	return 1;
}

static int
commonllp64(int unused, Fhdr *fp, ExecHdr *hp)
{
	long pgsize;
	uvlong entry;

	hswal(&hp->e, sizeof(Exec)/sizeof(long), beswal);
	if(!(hp->e.exechdr.magic & HDR_MAGIC))
		return 0;

	/*
	 * There can be more magic here if the
	 * header ever needs more expansion.
	 * For now just catch use of any of the
	 * unused bits.
	 */
	if((hp->e.exechdr.magic & ~DYN_MAGIC)>>16)
		return 0;
	entry = beswav(*(uvlong*)&hp->e.exechdr);

	pgsize = mach->pgsize;
	settext(fp, entry, pgsize+fp->hdrsz, hp->e.exechdr.text, fp->hdrsz);
	setdata(fp, _round(pgsize+fp->txtsz+fp->hdrsz, pgsize),
		hp->e.exechdr.data, fp->txtsz+fp->hdrsz, hp->e.exechdr.bss);
	setsym(fp, hp->e.exechdr.syms, hp->e.exechdr.spsz, hp->e.exechdr.pcsz, fp->datoff+fp->datsz);

	if(hp->e.exechdr.magic & DYN_MAGIC) {
		fp->txtaddr = 0;
		fp->dataddr = fp->txtsz;
		return 1;
	}
	commonboot(fp);
	return 1;
}

/*
 *	mips bootable image.
 */
static int
mipsboot(int fd, Fhdr *fp, ExecHdr *hp)
{
abort();
#ifdef unused
	USED(fd);
	fp->type = FMIPSB;
	switch(hp->e.exechdr.amagic) {
	default:
	case 0407:	/* some kind of mips */
		settext(fp, (u32int)hp->e.mentry, (u32int)hp->e.text_start,
			hp->e.tsize, sizeof(struct mipsexec)+4);
		setdata(fp, (u32int)hp->e.data_start, hp->e.dsize,
			fp->txtoff+hp->e.tsize, hp->e.bsize);
		break;
	case 0413:	/* some kind of mips */
		settext(fp, (u32int)hp->e.mentry, (u32int)hp->e.text_start,
			hp->e.tsize, 0);
		setdata(fp, (u32int)hp->e.data_start, hp->e.dsize,
			hp->e.tsize, hp->e.bsize);
		break;
	}
	setsym(fp, hp->e.nsyms, 0, hp->e.pcsize, hp->e.symptr);
	fp->hdrsz = 0;			/* header stripped */
#endif
	return 1;
}

/*
 *	mips4k bootable image.
 */
static int
mips4kboot(int fd, Fhdr *fp, ExecHdr *hp)
{
abort();
#ifdef unused
	USED(fd);
	fp->type = FMIPSB;
	switch(hp->e.h.amagic) {
	default:
	case 0407:	/* some kind of mips */
		settext(fp, (u32int)hp->e.h.mentry, (u32int)hp->e.h.text_start,
			hp->e.h.tsize, sizeof(struct mips4kexec));
		setdata(fp, (u32int)hp->e.h.data_start, hp->e.h.dsize,
			fp->txtoff+hp->e.h.tsize, hp->e.h.bsize);
		break;
	case 0413:	/* some kind of mips */
		settext(fp, (u32int)hp->e.h.mentry, (u32int)hp->e.h.text_start,
			hp->e.h.tsize, 0);
		setdata(fp, (u32int)hp->e.h.data_start, hp->e.h.dsize,
			hp->e.h.tsize, hp->e.h.bsize);
		break;
	}
	setsym(fp, hp->e.h.nsyms, 0, hp->e.h.pcsize, hp->e.h.symptr);
	fp->hdrsz = 0;			/* header stripped */
#endif
	return 1;
}

/*
 *	sparc bootable image
 */
static int
sparcboot(int fd, Fhdr *fp, ExecHdr *hp)
{
abort();
#ifdef unused
	USED(fd);
	fp->type = FSPARCB;
	settext(fp, hp->e.sentry, hp->e.sentry, hp->e.stext,
		sizeof(struct sparcexec));
	setdata(fp, hp->e.sentry+hp->e.stext, hp->e.sdata,
		fp->txtoff+hp->e.stext, hp->e.sbss);
	setsym(fp, hp->e.ssyms, 0, hp->e.sdrsize, fp->datoff+hp->e.sdata);
	fp->hdrsz = 0;			/* header stripped */
#endif
	return 1;
}

/*
 *	next bootable image
 */
static int
nextboot(int fd, Fhdr *fp, ExecHdr *hp)
{
abort();
#ifdef unused
	USED(fd);
	fp->type = FNEXTB;
	settext(fp, hp->e.textc.vmaddr, hp->e.textc.vmaddr,
		hp->e.texts.size, hp->e.texts.offset);
	setdata(fp, hp->e.datac.vmaddr, hp->e.datas.size,
		hp->e.datas.offset, hp->e.bsss.size);
	setsym(fp, hp->e.symc.nsyms, hp->e.symc.spoff, hp->e.symc.pcoff,
		hp->e.symc.symoff);
	fp->hdrsz = 0;			/* header stripped */
#endif
	return 1;
}

/*
 * Elf32 and Elf64 binaries.
 */
static int
elf64dotout(int fd, Fhdr *fp, ExecHdr *hp)
{

	uvlong (*swav)(uvlong);
	ulong (*swal)(ulong);
	ushort (*swab)(ushort);
	Ehdr64 *ep;
	Phdr64 *ph;
	int i, it, id, is, phsz;

	/* bitswap the header according to the DATA format */
	ep = &hp->e.elfhdr64;
	if(ep->ident[CLASS] != ELFCLASS64) {
		werrstr("bad ELF class - not 32 bit or 64 bit");
		return 0;
	}
	if(ep->ident[DATA] == ELFDATA2LSB) {
		swab = leswab;
		swal = leswal;
		swav = leswav;
	} else if(ep->ident[DATA] == ELFDATA2MSB) {
		swab = beswab;
		swal = beswal;
		swav = beswav;
	} else {
		werrstr("bad ELF encoding - not big or little endian");
		return 0;
	}

	ep->type = swab(ep->type);
	ep->machine = swab(ep->machine);
	ep->version = swal(ep->version);
	ep->elfentry = swal(ep->elfentry);
print("entry: 0x%x\n", ep->elfentry);
	ep->phoff = swav(ep->phoff);
	ep->shoff = swav(ep->shoff);
	ep->flags = swav(ep->flags);
	ep->ehsize = swab(ep->ehsize);
	ep->phentsize = swab(ep->phentsize);
	ep->phnum = swab(ep->phnum);
	ep->shentsize = swab(ep->shentsize);
	ep->shnum = swab(ep->shnum);
	ep->shstrndx = swab(ep->shstrndx);
	if(ep->type != EXEC || ep->version != CURRENT)
		return 0;

	/* we could definitely support a lot more machines here */
	fp->magic = ELF_MAG;
	fp->hdrsz = (ep->ehsize+ep->phnum*ep->phentsize+16)&~15;
	switch(ep->machine) {
	case I386:
		mach = &mi386;
		fp->type = FI386;
		break;
	case MIPS:
		mach = &mmips;
		fp->type = FMIPS;
		break;
	case SPARC64:
		mach = &msparc64;
		fp->type = FSPARC64;
		break;
	case POWER:
		mach = &mpower;
		fp->type = FPOWER;
		break;
	case AMD64:
		mach = &mamd64;
		fp->type = FAMD64;
		break;
	case ARM:
		mach = &marm;
		fp->type = FARM;
		break;
	default:
		return 0;
	}

	if(ep->phentsize != sizeof(Phdr64)) {
		werrstr("bad ELF header size");
		return 0;
	}
	phsz = sizeof(Phdr64)*ep->phnum;
	ph = malloc(phsz);
	if(!ph)
		return 0;
	seek(fd, ep->phoff, 0);
	if(read(fd, ph, phsz) < 0) {
		free(ph);
		return 0;
	}
	hswal(ph, phsz/sizeof(ulong), swal);

	/* find text, data and symbols and install them */
	it = id = is = -1;
	for(i = 0; i < ep->phnum; i++) {
		if(ph[i].type == LOAD
		&& (ph[i].flags & (R|X)) == (R|X) && it == -1)
			it = i;
		else if(ph[i].type == LOAD
		&& (ph[i].flags & (R|W)) == (R|W) && id == -1)
			id = i;
		else if(ph[i].type == NOPTYPE && is == -1)
			is = i;
	}
	if(it == -1 || id == -1) {
		/*
		 * The SPARC64 boot image is something of an ELF hack.
		 * Text+Data+BSS are represented by ph[0].  Symbols
		 * are represented by ph[1]:
		 *
		 *		filesz, memsz, vaddr, paddr, off
		 * ph[0] : txtsz+datsz, txtsz+datsz+bsssz, txtaddr-KZERO, datasize, txtoff
		 * ph[1] : symsz, lcsz, 0, 0, symoff
		 */
		if(ep->machine == SPARC64 && ep->phnum == 2) {
			ulong txtaddr, txtsz, dataddr, bsssz;

			txtaddr = ph[0].vaddr | 0x80000000;
			txtsz = ph[0].filesz - ph[0].paddr;
			dataddr = txtaddr + txtsz;
			bsssz = ph[0].memsz - ph[0].filesz;
			settext(fp, ep->elfentry | 0x80000000, txtaddr, txtsz, ph[0].offset);
			setdata(fp, dataddr, ph[0].paddr, ph[0].offset + txtsz, bsssz);
			setsym(fp, ph[1].filesz, 0, ph[1].memsz, ph[1].offset);
			free(ph);
			return 1;
		}

		werrstr("No TEXT or DATA sections");
		free(ph);
		return 0;
	}

	settext(fp, ep->elfentry, ph[it].vaddr, ph[it].memsz, ph[it].offset);
	setdata(fp, ph[id].vaddr, ph[id].filesz, ph[id].offset, ph[id].memsz - ph[id].filesz);
	if(is != -1)
		setsym(fp, ph[is].filesz, 0, ph[is].memsz, ph[is].offset);
	free(ph);
	return 1;
}

static int
elfdotout(int fd, Fhdr *fp, ExecHdr *hp)
{

	ulong (*swal)(ulong);
	ushort (*swab)(ushort);
	Ehdr32 *ep;
	Phdr32 *ph;
	int i, it, id, is, phsz;

	/* bitswap the header according to the DATA format */
	ep = &hp->e.elfhdr32;
	if(ep->ident[CLASS] != ELFCLASS32) {
		return elf64dotout(fd, fp, hp);
	}
	if(ep->ident[DATA] == ELFDATA2LSB) {
		swab = leswab;
		swal = leswal;
	} else if(ep->ident[DATA] == ELFDATA2MSB) {
		swab = beswab;
		swal = beswal;
	} else {
		werrstr("bad ELF encoding - not big or little endian");
		return 0;
	}

	ep->type = swab(ep->type);
	ep->machine = swab(ep->machine);
	ep->version = swal(ep->version);
	ep->elfentry = swal(ep->elfentry);
	ep->phoff = swal(ep->phoff);
	ep->shoff = swal(ep->shoff);
	ep->flags = swal(ep->flags);
	ep->ehsize = swab(ep->ehsize);
	ep->phentsize = swab(ep->phentsize);
	ep->phnum = swab(ep->phnum);
	ep->shentsize = swab(ep->shentsize);
	ep->shnum = swab(ep->shnum);
	ep->shstrndx = swab(ep->shstrndx);
	if(ep->type != EXEC || ep->version != CURRENT)
		return 0;

	/* we could definitely support a lot more machines here */
	fp->magic = ELF_MAG;
	fp->hdrsz = (ep->ehsize+ep->phnum*ep->phentsize+16)&~15;
	switch(ep->machine) {
	case I386:
		mach = &mi386;
		fp->type = FI386;
		break;
	case MIPS:
		mach = &mmips;
		fp->type = FMIPS;
		break;
	case SPARC64:
		mach = &msparc64;
		fp->type = FSPARC64;
		break;
	case POWER:
		mach = &mpower;
		fp->type = FPOWER;
		break;
	case AMD64:
		mach = &mamd64;
		fp->type = FAMD64;
		break;
	case ARM:
		mach = &marm;
		fp->type = FARM;
		break;
	default:
		return 0;
	}

	if(ep->phentsize != sizeof(Phdr32)) {
		werrstr("bad ELF header size");
		return 0;
	}
	phsz = sizeof(Phdr32)*ep->phnum;
	ph = malloc(phsz);
	if(!ph)
		return 0;
	seek(fd, ep->phoff, 0);
	if(read(fd, ph, phsz) < 0) {
		free(ph);
		return 0;
	}
	hswal(ph, phsz/sizeof(ulong), swal);

	/* find text, data and symbols and install them */
	it = id = is = -1;
	for(i = 0; i < ep->phnum; i++) {
		if(ph[i].type == LOAD
		&& (ph[i].flags & (R|X)) == (R|X) && it == -1)
			it = i;
		else if(ph[i].type == LOAD
		&& (ph[i].flags & (R|W)) == (R|W) && id == -1)
			id = i;
		else if(ph[i].type == NOPTYPE && is == -1)
			is = i;
	}
	if(it == -1 || id == -1) {
		/*
		 * The SPARC64 boot image is something of an ELF hack.
		 * Text+Data+BSS are represented by ph[0].  Symbols
		 * are represented by ph[1]:
		 *
		 *		filesz, memsz, vaddr, paddr, off
		 * ph[0] : txtsz+datsz, txtsz+datsz+bsssz, txtaddr-KZERO, datasize, txtoff
		 * ph[1] : symsz, lcsz, 0, 0, symoff
		 */
		if(ep->machine == SPARC64 && ep->phnum == 2) {
			ulong txtaddr, txtsz, dataddr, bsssz;

			txtaddr = ph[0].vaddr | 0x80000000;
			txtsz = ph[0].filesz - ph[0].paddr;
			dataddr = txtaddr + txtsz;
			bsssz = ph[0].memsz - ph[0].filesz;
			settext(fp, ep->elfentry | 0x80000000, txtaddr, txtsz, ph[0].offset);
			setdata(fp, dataddr, ph[0].paddr, ph[0].offset + txtsz, bsssz);
			setsym(fp, ph[1].filesz, 0, ph[1].memsz, ph[1].offset);
			free(ph);
			return 1;
		}

		werrstr("No TEXT or DATA sections");
		free(ph);
		return 0;
	}

	settext(fp, ep->elfentry, ph[it].vaddr, ph[it].memsz, ph[it].offset);
	setdata(fp, ph[id].vaddr, ph[id].filesz, ph[id].offset, ph[id].memsz - ph[id].filesz);
	if(is != -1)
		setsym(fp, ph[is].filesz, 0, ph[is].memsz, ph[is].offset);
	free(ph);
	return 1;
}

/*
 * (Free|Net)BSD ARM header.
 */
static int
armdotout(int fd, Fhdr *fp, ExecHdr *hp)
{
	uvlong kbase;

	USED(fd);
	settext(fp, hp->e.exechdr.entry, sizeof(Exec), hp->e.exechdr.text, sizeof(Exec));
	setdata(fp, fp->txtsz, hp->e.exechdr.data, fp->txtsz, hp->e.exechdr.bss);
	setsym(fp, hp->e.exechdr.syms, hp->e.exechdr.spsz, hp->e.exechdr.pcsz, fp->datoff+fp->datsz);

	kbase = 0xF0000000;
	if ((fp->entry & kbase) == kbase) {		/* Boot image */
		fp->txtaddr = kbase+sizeof(Exec);
		fp->name = "ARM *BSD boot image";
		fp->hdrsz = 0;		/* header stripped */
		fp->dataddr = kbase+fp->txtsz;
	}
	return 1;
}

static void
settext(Fhdr *fp, uvlong e, uvlong a, long s, vlong off)
{
	fp->txtaddr = a;
	fp->entry = e;
	fp->txtsz = s;
	fp->txtoff = off;
}

static void
setdata(Fhdr *fp, uvlong a, long s, vlong off, long bss)
{
	fp->dataddr = a;
	fp->datsz = s;
	fp->datoff = off;
	fp->bsssz = bss;
}

static void
setsym(Fhdr *fp, long symsz, long sppcsz, long lnpcsz, vlong symoff)
{
	fp->symsz = symsz;
	fp->symoff = symoff;
	fp->sppcsz = sppcsz;
	fp->sppcoff = fp->symoff+fp->symsz;
	fp->lnpcsz = lnpcsz;
	fp->lnpcoff = fp->sppcoff+fp->sppcsz;
}


static uvlong
_round(uvlong a, ulong b)
{
	uvlong w;

	w = (a/b)*b;
	if (a!=w)
		w += b;
	return(w);
}
