// Inferno libmach/sym.c
// http://code.google.com/p/inferno-os/source/browse/utils/libmach/sym.c
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

#include <u.h>
#include <libc.h>
#include <bio.h>
#include <mach.h>

#define	HUGEINT	0x7fffffff
#define	NNAME	20		/* a relic of the past */

typedef	struct txtsym Txtsym;
typedef	struct file File;
typedef	struct hist Hist;

struct txtsym {				/* Text Symbol table */
	int 	n;			/* number of local vars */
	Sym	**locals;		/* array of ptrs to autos */
	Sym	*sym;			/* function symbol entry */
};

struct hist {				/* Stack of include files & #line directives */
	char	*name;			/* Assumes names Null terminated in file */
	int32	line;			/* line # where it was included */
	int32	offset;			/* line # of #line directive */
};

struct file {				/* Per input file header to history stack */
	uvlong	addr;			/* address of first text sym */
	union {
		Txtsym	*txt;		/* first text symbol */
		Sym	*sym;		/* only during initilization */
	};
	int	n;			/* size of history stack */
	Hist	*hist;			/* history stack */
};

static	int	debug = 0;

static	Sym	**autos;		/* Base of auto variables */
static	File	*files;			/* Base of file arena */
static	int	fmaxi;			/* largest file path index */
static	Sym	**fnames;		/* file names path component table */
static	Sym	**globals;		/* globals by addr table */
static	Hist	*hist;			/* base of history stack */
static	int	isbuilt;		/* internal table init flag */
static	int32	nauto;			/* number of automatics */
static	int32	nfiles;			/* number of files */
static	int32	nglob;			/* number of globals */
static	int32	nhist;			/* number of history stack entries */
static	int32	nsym;			/* number of symbols */
static	int	ntxt;			/* number of text symbols */
static	uchar	*pcline;		/* start of pc-line state table */
static	uchar 	*pclineend;		/* end of pc-line table */
static	uchar	*spoff;			/* start of pc-sp state table */
static	uchar	*spoffend;		/* end of pc-sp offset table */
static	Sym	*symbols;		/* symbol table */
static	Txtsym	*txt;			/* Base of text symbol table */
static	uvlong	txtstart;		/* start of text segment */
static	uvlong	txtend;			/* end of text segment */
static	uvlong	firstinstr;		/* as found from symtab; needed for amd64 */

static void	cleansyms(void);
static int32	decodename(Biobuf*, Sym*);
static short	*encfname(char*);
static int 	fline(char*, int, int32, Hist*, Hist**);
static void	fillsym(Sym*, Symbol*);
static int	findglobal(char*, Symbol*);
static int	findlocvar(Symbol*, char *, Symbol*);
static int	findtext(char*, Symbol*);
static int	hcomp(Hist*, short*);
static int	hline(File*, short*, int32*);
static void	printhist(char*, Hist*, int);
static int	buildtbls(void);
static int	symcomp(const void*, const void*);
static int	symerrmsg(int, char*);
static int	txtcomp(const void*, const void*);
static int	filecomp(const void*, const void*);

/*
 *	initialize the symbol tables
 */
int
syminit(int fd, Fhdr *fp)
{
	Sym *p;
	int32 i, l, size, symsz;
	vlong vl;
	Biobuf b;
	int svalsz, newformat, shift;
	uvlong (*swav)(uvlong);
	uint32 (*swal)(uint32);
	uchar buf[8], c;

	if(fp->symsz == 0)
		return 0;
	if(fp->type == FNONE)
		return 0;

	swav = beswav;
	swal = beswal;

	cleansyms();
	textseg(fp->txtaddr, fp);
		/* minimum symbol record size = 4+1+2 bytes */
	symbols = malloc((fp->symsz/(4+1+2)+1)*sizeof(Sym));
	if(symbols == 0) {
		werrstr("can't malloc %d bytes", fp->symsz);
		return -1;
	}
	Binit(&b, fd, OREAD);
	Bseek(&b, fp->symoff, 0);
	memset(buf, 0, sizeof buf);
	Bread(&b, buf, sizeof buf);
	newformat = 0;
	symsz = fp->symsz;
	if(memcmp(buf, "\xfd\xff\xff\xff\x00\x00\x00", 7) == 0) {
		swav = leswav;
		swal = leswal;
		newformat = 1;
	} else if(memcmp(buf, "\xff\xff\xff\xfd\x00\x00\x00", 7) == 0) {
		newformat = 1;
	} else if(memcmp(buf, "\xfe\xff\xff\xff\x00\x00", 6) == 0) {
		// Table format used between Go 1.0 and Go 1.1:
		// little-endian but otherwise same as the old Go 1.0 table.
		// Not likely to be seen much in practice, but easy to handle.
		swav = leswav;
		swal = leswal;
		Bseek(&b, fp->symoff+6, 0);
		symsz -= 6;
	} else {
		Bseek(&b, fp->symoff, 0);
	}
	svalsz = 0;
	if(newformat) {
		svalsz = buf[7];
		if(svalsz != 4 && svalsz != 8) {
			werrstr("invalid word size %d bytes", svalsz);
			return -1;
		}
		symsz -= 8;
	}

	nsym = 0;
	size = 0;
	for(p = symbols; size < symsz; p++, nsym++) {
		if(newformat) {
			// Go 1.1 format. See comment at top of ../pkg/runtime/symtab.c.
			if(Bread(&b, &c, 1) != 1)
				return symerrmsg(1, "symbol");
			if((c&0x3F) < 26)
				p->type = (c&0x3F)+ 'A';
			else
				p->type = (c&0x3F) - 26 + 'a';
			size++;

			if(c&0x40) {
				// Fixed-width address.
				if(svalsz == 8) {
					if(Bread(&b, &vl, 8) != 8)
						return symerrmsg(8, "symbol");
					p->value = swav(vl);
				} else {
					if(Bread(&b, &l, 4) != 4)
						return symerrmsg(4, "symbol");
					p->value = (u32int)swal(l);
				}
				size += svalsz;
			} else {
				// Varint address.
				shift = 0;
				p->value = 0;
				for(;;) {
					if(Bread(&b, buf, 1) != 1)
						return symerrmsg(1, "symbol");
					p->value |= (uint64)(buf[0]&0x7F)<<shift;
					shift += 7;
					size++;
					if((buf[0]&0x80) == 0)
						break;
				}
			}
			p->gotype = 0;
			if(c&0x80) {
				// Has Go type. Fixed-width address.
				if(svalsz == 8) {
					if(Bread(&b, &vl, 8) != 8)
						return symerrmsg(8, "symbol");
					p->gotype = swav(vl);
				} else {
					if(Bread(&b, &l, 4) != 4)
						return symerrmsg(4, "symbol");
					p->gotype = (u32int)swal(l);
				}
				size += svalsz;
			}
			
			// Name.
			p->type |= 0x80; // for decodename
			i = decodename(&b, p);
			if(i < 0)
				return -1;
			size += i;
		} else {
			// Go 1.0 format: Plan 9 format + go type symbol.
			if(fp->_magic && (fp->magic & HDR_MAGIC)){
				svalsz = 8;
				if(Bread(&b, &vl, 8) != 8)
					return symerrmsg(8, "symbol");
				p->value = swav(vl);
			}
			else{
				svalsz = 4;
				if(Bread(&b, &l, 4) != 4)
					return symerrmsg(4, "symbol");
				p->value = (u32int)swal(l);
			}
			if(Bread(&b, &p->type, sizeof(p->type)) != sizeof(p->type))
				return symerrmsg(sizeof(p->value), "symbol");
	
			i = decodename(&b, p);
			if(i < 0)
				return -1;
			size += i+svalsz+sizeof(p->type);
	
			if(svalsz == 8){
				if(Bread(&b, &vl, 8) != 8)
					return symerrmsg(8, "symbol");
				p->gotype = swav(vl);
			}
			else{
				if(Bread(&b, &l, 4) != 4)
					return symerrmsg(4, "symbol");
				p->gotype = (u32int)swal(l);
			}
			size += svalsz;
		}

		/* count global & auto vars, text symbols, and file names */
		switch (p->type) {
		case 'l':
		case 'L':
		case 't':
		case 'T':
			ntxt++;
			break;
		case 'd':
		case 'D':
		case 'b':
		case 'B':
			nglob++;
			break;
		case 'f':
			if(strcmp(p->name, ".frame") == 0) {
				p->type = 'm';
				nauto++;
			}
			else if(p->value > fmaxi)
				fmaxi = p->value;	/* highest path index */
			break;
		case 'a':
		case 'p':
		case 'm':
			nauto++;
			break;
		case 'z':
			if(p->value == 1) {		/* one extra per file */
				nhist++;
				nfiles++;
			}
			nhist++;
			break;
		default:
			break;
		}
	}
	if (debug)
		print("NG: %d NT: %d NF: %d\n", nglob, ntxt, fmaxi);
	if (fp->sppcsz) {			/* pc-sp offset table */
		spoff = (uchar *)malloc(fp->sppcsz);
		if(spoff == 0) {
			werrstr("can't malloc %d bytes", fp->sppcsz);
			return -1;
		}
		Bseek(&b, fp->sppcoff, 0);
		if(Bread(&b, spoff, fp->sppcsz) != fp->sppcsz){
			spoff = 0;
			return symerrmsg(fp->sppcsz, "sp-pc");
		}
		spoffend = spoff+fp->sppcsz;
	}
	if (fp->lnpcsz) {			/* pc-line number table */
		pcline = (uchar *)malloc(fp->lnpcsz);
		if(pcline == 0) {
			werrstr("can't malloc %d bytes", fp->lnpcsz);
			return -1;
		}
		Bseek(&b, fp->lnpcoff, 0);
		if(Bread(&b, pcline, fp->lnpcsz) != fp->lnpcsz){
			pcline = 0;
			return symerrmsg(fp->lnpcsz, "pc-line");
		}
		pclineend = pcline+fp->lnpcsz;
	}
	return nsym;
}

static int
symerrmsg(int n, char *table)
{
	werrstr("can't read %d bytes of %s table", n, table);
	return -1;
}

static int32
decodename(Biobuf *bp, Sym *p)
{
	char *cp;
	int c1, c2;
	int32 n;
	vlong o;

	if((p->type & 0x80) == 0) {		/* old-style, fixed length names */
		p->name = malloc(NNAME);
		if(p->name == 0) {
			werrstr("can't malloc %d bytes", NNAME);
			return -1;
		}
		if(Bread(bp, p->name, NNAME) != NNAME)
			return symerrmsg(NNAME, "symbol");
		Bseek(bp, 3, 1);
		return NNAME+3;
	}

	p->type &= ~0x80;
	if(p->type == 'z' || p->type == 'Z') {
		o = Bseek(bp, 0, 1);
		if(Bgetc(bp) < 0) {
			werrstr("can't read symbol name");
			return -1;
		}
		for(;;) {
			c1 = Bgetc(bp);
			c2 = Bgetc(bp);
			if(c1 < 0 || c2 < 0) {
				werrstr("can't read symbol name");
				return -1;
			}
			if(c1 == 0 && c2 == 0)
				break;
		}
		n = Bseek(bp, 0, 1)-o;
		p->name = malloc(n);
		if(p->name == 0) {
			werrstr("can't malloc %d bytes", n);
			return -1;
		}
		Bseek(bp, -n, 1);
		if(Bread(bp, p->name, n) != n) {
			werrstr("can't read %d bytes of symbol name", n);
			return -1;
		}
	} else {
		cp = Brdline(bp, '\0');
		if(cp == 0) {
			werrstr("can't read symbol name");
			return -1;
		}
		n = Blinelen(bp);
		p->name = malloc(n);
		if(p->name == 0) {
			werrstr("can't malloc %d bytes", n);
			return -1;
		}
		strcpy(p->name, cp);
	}
	return n;
}

/*
 *	free any previously loaded symbol tables
 */
static void
cleansyms(void)
{
	if(globals)
		free(globals);
	globals = 0;
	nglob = 0;
	if(txt)
		free(txt);
	txt = 0;
	ntxt = 0;
	if(fnames)
		free(fnames);
	fnames = 0;
	fmaxi = 0;

	if(files)
		free(files);
	files = 0;
	nfiles = 0;
	if(hist)
		free(hist);
	hist = 0;
	nhist = 0;
	if(autos)
		free(autos);
	autos = 0;
	nauto = 0;
	isbuilt = 0;
	if(symbols)
		free(symbols);
	symbols = 0;
	nsym = 0;
	if(spoff)
		free(spoff);
	spoff = 0;
	if(pcline)
		free(pcline);
	pcline = 0;
}

/*
 *	delimit the text segment
 */
void
textseg(uvlong base, Fhdr *fp)
{
	txtstart = base;
	txtend = base+fp->txtsz;
}

/*
 *	symbase: return base and size of raw symbol table
 *		(special hack for high access rate operations)
 */
Sym *
symbase(int32 *n)
{
	*n = nsym;
	return symbols;
}

/*
 *	Get the ith symbol table entry
 */
Sym *
getsym(int index)
{
	if(index >= 0 && index < nsym)
		return &symbols[index];
	return 0;
}

/*
 *	initialize internal symbol tables
 */
static int
buildtbls(void)
{
	int32 i;
	int j, nh, ng, nt;
	File *f;
	Txtsym *tp;
	Hist *hp;
	Sym *p, **ap;

	if(isbuilt)
		return 1;
	isbuilt = 1;
			/* allocate the tables */
	firstinstr = 0;
	if(nglob) {
		globals = malloc(nglob*sizeof(*globals));
		if(!globals) {
			werrstr("can't malloc global symbol table");
			return 0;
		}
	}
	if(ntxt) {
		txt = malloc(ntxt*sizeof(*txt));
		if (!txt) {
			werrstr("can't malloc text symbol table");
			return 0;
		}
	}
	fnames = malloc((fmaxi+1)*sizeof(*fnames));
	if (!fnames) {
		werrstr("can't malloc file name table");
		return 0;
	}
	memset(fnames, 0, (fmaxi+1)*sizeof(*fnames));
	files = malloc(nfiles*sizeof(*files));
	if(!files) {
		werrstr("can't malloc file table");
		return 0;
	}
	hist = malloc(nhist*sizeof(Hist));
	if(hist == 0) {
		werrstr("can't malloc history stack");
		return 0;
	}
	autos = malloc(nauto*sizeof(Sym*));
	if(autos == 0) {
		werrstr("can't malloc auto symbol table");
		return 0;
	}
		/* load the tables */
	ng = nt = nh = 0;
	f = 0;
	tp = 0;
	i = nsym;
	hp = hist;
	ap = autos;
	for(p = symbols; i-- > 0; p++) {
//print("sym %d type %c name %s value %llux\n", p-symbols, p->type, p->name, p->value);
		switch(p->type) {
		case 'D':
		case 'd':
		case 'B':
		case 'b':
			if(debug)
				print("Global: %s %llux\n", p->name, p->value);
			globals[ng++] = p;
			break;
		case 'z':
			if(p->value == 1) {		/* New file */
				if(f) {
					f->n = nh;
					f->hist[nh].name = 0;	/* one extra */
					hp += nh+1;
					f++;
				}
				else
					f = files;
				f->hist = hp;
				f->sym = 0;
				f->addr = 0;
				nh = 0;
			}
				/* alloc one slot extra as terminator */
			f->hist[nh].name = p->name;
			f->hist[nh].line = p->value;
			f->hist[nh].offset = 0;
			if(debug)
				printhist("-> ", &f->hist[nh], 1);
			nh++;
			break;
		case 'Z':
			if(f && nh > 0)
				f->hist[nh-1].offset = p->value;
			break;
		case 'T':
		case 't':	/* Text: terminate history if first in file */
		case 'L':
		case 'l':
			tp = &txt[nt++];
			tp->n = 0;
			tp->sym = p;
			tp->locals = ap;
			if(debug)
				print("TEXT: %s at %llux\n", p->name, p->value);
			if (firstinstr == 0 || p->value < firstinstr)
				firstinstr = p->value;
			if(f && !f->sym) {			/* first  */
				f->sym = p;
				f->addr = p->value;
			}
			break;
		case 'a':
		case 'p':
		case 'm':		/* Local Vars */
			if(!tp)
				print("Warning: Free floating local var: %s\n",
					p->name);
			else {
				if(debug)
					print("Local: %s %llux\n", p->name, p->value);
				tp->locals[tp->n] = p;
				tp->n++;
				ap++;
			}
			break;
		case 'f':		/* File names */
			if(debug)
				print("Fname: %s\n", p->name);
			fnames[p->value] = p;
			break;
		default:
			break;
		}
	}
		/* sort global and text tables into ascending address order */
	qsort(globals, nglob, sizeof(Sym*), symcomp);
	qsort(txt, ntxt, sizeof(Txtsym), txtcomp);
	qsort(files, nfiles, sizeof(File), filecomp);
	tp = txt;
	for(i = 0, f = files; i < nfiles; i++, f++) {
		for(j = 0; j < ntxt; j++) {
			if(f->sym == tp->sym) {
				if(debug) {
					print("LINK: %s to at %llux", f->sym->name, f->addr);
					printhist("... ", f->hist, 1);
				}
				f->txt = tp++;
				break;
			}
			if(++tp >= txt+ntxt)	/* wrap around */
				tp = txt;
		}
	}
	return 1;
}

/*
 * find symbol function.var by name.
 *	fn != 0 && var != 0	=> look for fn in text, var in data
 *	fn != 0 && var == 0	=> look for fn in text
 *	fn == 0 && var != 0	=> look for var first in text then in data space.
 */
int
lookup(char *fn, char *var, Symbol *s)
{
	int found;

	if(buildtbls() == 0)
		return 0;
	if(fn) {
		found = findtext(fn, s);
		if(var == 0)		/* case 2: fn not in text */
			return found;
		else if(!found)		/* case 1: fn not found */
			return 0;
	} else if(var) {
		found = findtext(var, s);
		if(found)
			return 1;	/* case 3: var found in text */
	} else return 0;		/* case 4: fn & var == zero */

	if(found)
		return findlocal(s, var, s);	/* case 1: fn found */
	return findglobal(var, s);		/* case 3: var not found */

}

/*
 * strcmp, but allow '_' to match center dot (rune 00b7 == bytes c2 b7)
 */
int
cdotstrcmp(char *sym, char *user)
{
	for (;;) {
		while (*sym == *user) {
			if (*sym++ == '\0')
				return 0;
			user++;
		}
		/* unequal - but maybe '_' matches center dot */
		if (user[0] == '_' && (sym[0]&0xFF) == 0xc2 && (sym[1]&0xFF) == 0xb7) {
			/* '_' matches center dot - advance and continue */
			user++;
			sym += 2;
			continue;
		}
		break;
	}
	return *user - *sym;
}

/*
 * find a function by name
 */
static int
findtext(char *name, Symbol *s)
{
	int i;

	for(i = 0; i < ntxt; i++) {
		if(cdotstrcmp(txt[i].sym->name, name) == 0) {
			fillsym(txt[i].sym, s);
			s->handle = (void *) &txt[i];
			s->index = i;
			return 1;
		}
	}
	return 0;
}
/*
 * find global variable by name
 */
static int
findglobal(char *name, Symbol *s)
{
	int32 i;

	for(i = 0; i < nglob; i++) {
		if(cdotstrcmp(globals[i]->name, name) == 0) {
			fillsym(globals[i], s);
			s->index = i;
			return 1;
		}
	}
	return 0;
}

/*
 *	find the local variable by name within a given function
 */
int
findlocal(Symbol *s1, char *name, Symbol *s2)
{
	if(s1 == 0)
		return 0;
	if(buildtbls() == 0)
		return 0;
	return findlocvar(s1, name, s2);
}

/*
 *	find the local variable by name within a given function
 *		(internal function - does no parameter validation)
 */
static int
findlocvar(Symbol *s1, char *name, Symbol *s2)
{
	Txtsym *tp;
	int i;

	tp = (Txtsym *)s1->handle;
	if(tp && tp->locals) {
		for(i = 0; i < tp->n; i++)
			if (cdotstrcmp(tp->locals[i]->name, name) == 0) {
				fillsym(tp->locals[i], s2);
				s2->handle = (void *)tp;
				s2->index = tp->n-1 - i;
				return 1;
			}
	}
	return 0;
}

/*
 *	Get ith text symbol
 */
int
textsym(Symbol *s, int index)
{

	if(buildtbls() == 0)
		return 0;
	if(index < 0 || index >= ntxt)
		return 0;
	fillsym(txt[index].sym, s);
	s->handle = (void *)&txt[index];
	s->index = index;
	return 1;
}

/*
 *	Get ith file name
 */
int
filesym(int index, char *buf, int n)
{
	Hist *hp;

	if(buildtbls() == 0)
		return 0;
	if(index < 0 || index >= nfiles)
		return 0;
	hp = files[index].hist;
	if(!hp || !hp->name)
		return 0;
	return fileelem(fnames, (uchar*)hp->name, buf, n);
}

/*
 *	Lookup name of local variable located at an offset into the frame.
 *	The type selects either a parameter or automatic.
 */
int
getauto(Symbol *s1, int off, int type, Symbol *s2)
{
	Txtsym *tp;
	Sym *p;
	int i, t;

	if(s1 == 0)
		return 0;
	if(type == CPARAM)
		t = 'p';
	else if(type == CAUTO)
		t = 'a';
	else
		return 0;
	if(buildtbls() == 0)
		return 0;
	tp = (Txtsym *)s1->handle;
	if(tp == 0)
		return 0;
	for(i = 0; i < tp->n; i++) {
		p = tp->locals[i];
		if(p->type == t && p->value == off) {
			fillsym(p, s2);
			s2->handle = s1->handle;
			s2->index = tp->n-1 - i;
			return 1;
		}
	}
	return 0;
}

/*
 * Find text symbol containing addr; binary search assumes text array is sorted by addr
 */
static int
srchtext(uvlong addr)
{
	uvlong val;
	int top, bot, mid;
	Sym *sp;

	val = addr;
	bot = 0;
	top = ntxt;
	for (mid = (bot+top)/2; mid < top; mid = (bot+top)/2) {
		sp = txt[mid].sym;
		if(val < sp->value)
			top = mid;
		else if(mid != ntxt-1 && val >= txt[mid+1].sym->value)
			bot = mid;
		else
			return mid;
	}
	return -1;
}

/*
 * Find data symbol containing addr; binary search assumes data array is sorted by addr
 */
static int
srchdata(uvlong addr)
{
	uvlong val;
	int top, bot, mid;
	Sym *sp;

	bot = 0;
	top = nglob;
	val = addr;
	for(mid = (bot+top)/2; mid < top; mid = (bot+top)/2) {
		sp = globals[mid];
		if(val < sp->value)
			top = mid;
		else if(mid < nglob-1 && val >= globals[mid+1]->value)
			bot = mid;
		else
			return mid;
	}
	return -1;
}

/*
 * Find symbol containing val in specified search space
 * There is a special case when a value falls beyond the end
 * of the text segment; if the search space is CTEXT, that value
 * (usually etext) is returned.  If the search space is CANY, symbols in the
 * data space are searched for a match.
 */
int
findsym(uvlong val, int type, Symbol *s)
{
	int i;

	if(buildtbls() == 0)
		return 0;

	if(type == CTEXT || type == CANY) {
		i = srchtext(val);
		if(i >= 0) {
			if(type == CTEXT || i != ntxt-1) {
				fillsym(txt[i].sym, s);
				s->handle = (void *) &txt[i];
				s->index = i;
				return 1;
			}
		}
	}
	if(type == CDATA || type == CANY) {
		i = srchdata(val);
		if(i >= 0) {
			fillsym(globals[i], s);
			s->index = i;
			return 1;
		}
	}
	return 0;
}

/*
 *	Find the start and end address of the function containing addr
 */
int
fnbound(uvlong addr, uvlong *bounds)
{
	int i;

	if(buildtbls() == 0)
		return 0;

	i = srchtext(addr);
	if(0 <= i && i < ntxt-1) {
		bounds[0] = txt[i].sym->value;
		bounds[1] = txt[i+1].sym->value;
		return 1;
	}
	return 0;
}

/*
 * get the ith local symbol for a function
 * the input symbol table is reverse ordered, so we reverse
 * accesses here to maintain approx. parameter ordering in a stack trace.
 */
int
localsym(Symbol *s, int index)
{
	Txtsym *tp;

	if(s == 0 || index < 0)
		return 0;
	if(buildtbls() == 0)
		return 0;

	tp = (Txtsym *)s->handle;
	if(tp && tp->locals && index < tp->n) {
		fillsym(tp->locals[tp->n-index-1], s);	/* reverse */
		s->handle = (void *)tp;
		s->index = index;
		return 1;
	}
	return 0;
}

/*
 * get the ith global symbol
 */
int
globalsym(Symbol *s, int index)
{
	if(s == 0)
		return 0;
	if(buildtbls() == 0)
		return 0;

	if(index >=0 && index < nglob) {
		fillsym(globals[index], s);
		s->index = index;
		return 1;
	}
	return 0;
}

/*
 *	find the pc given a file name and line offset into it.
 */
uvlong
file2pc(char *file, int32 line)
{
	File *fp;
	int32 i;
	uvlong pc, start, end;
	short *name;

	if(buildtbls() == 0 || files == 0)
		return ~0;
	name = encfname(file);
	if(name == 0) {			/* encode the file name */
		werrstr("file %s not found", file);
		return ~0;
	}
		/* find this history stack */
	for(i = 0, fp = files; i < nfiles; i++, fp++)
		if (hline(fp, name, &line))
			break;
	free(name);
	if(i >= nfiles) {
		werrstr("line %d in file %s not found", line, file);
		return ~0;
	}
	start = fp->addr;		/* first text addr this file */
	if(i < nfiles-1)
		end = (fp+1)->addr;	/* first text addr next file */
	else
		end = 0;		/* last file in load module */
	/*
	 * At this point, line contains the offset into the file.
	 * run the state machine to locate the pc closest to that value.
	 */
	if(debug)
		print("find pc for %d - between: %llux and %llux\n", line, start, end);
	pc = line2addr(line, start, end);
	if(pc == ~0) {
		werrstr("line %d not in file %s", line, file);
		return ~0;
	}
	return pc;
}

/*
 *	search for a path component index
 */
static int
pathcomp(char *s, int n)
{
	int i;

	for(i = 0; i <= fmaxi; i++)
		if(fnames[i] && strncmp(s, fnames[i]->name, n) == 0)
			return i;
	return -1;
}

/*
 *	Encode a char file name as a sequence of short indices
 *	into the file name dictionary.
 */
static short*
encfname(char *file)
{
	int i, j;
	char *cp, *cp2;
	short *dest;

	if(*file == '/')	/* always check first '/' */
		cp2 = file+1;
	else {
		cp2 = strchr(file, '/');
		if(!cp2)
			cp2 = strchr(file, 0);
	}
	cp = file;
	dest = 0;
	for(i = 0; *cp; i++) {
		j = pathcomp(cp, cp2-cp);
		if(j < 0)
			return 0;	/* not found */
		dest = realloc(dest, (i+1)*sizeof(short));
		dest[i] = j;
		cp = cp2;
		while(*cp == '/')	/* skip embedded '/'s */
			cp++;
		cp2 = strchr(cp, '/');
		if(!cp2)
			cp2 = strchr(cp, 0);
	}
	dest = realloc(dest, (i+1)*sizeof(short));
	dest[i] = 0;
	return dest;
}

/*
 *	Search a history stack for a matching file name accumulating
 *	the size of intervening files in the stack.
 */
static int
hline(File *fp, short *name, int32 *line)
{
	Hist *hp;
	int offset, depth;
	int32 ln;

	for(hp = fp->hist; hp->name; hp++)		/* find name in stack */
		if(hp->name[1] || hp->name[2]) {
			if(hcomp(hp, name))
				break;
		}
	if(!hp->name)		/* match not found */
		return 0;
	if(debug)
		printhist("hline found ... ", hp, 1);
	/*
	 * unwind the stack until empty or we hit an entry beyond our line
	 */
	ln = *line;
	offset = hp->line-1;
	depth = 1;
	for(hp++; depth && hp->name; hp++) {
		if(debug)
			printhist("hline inspect ... ", hp, 1);
		if(hp->name[1] || hp->name[2]) {
			if(hp->offset){			/* Z record */
				offset = 0;
				if(hcomp(hp, name)) {
					if(*line <= hp->offset)
						break;
					ln = *line+hp->line-hp->offset;
					depth = 1;	/* implicit pop */
				} else
					depth = 2;	/* implicit push */
			} else if(depth == 1 && ln < hp->line-offset)
					break;		/* Beyond our line */
			else if(depth++ == 1)		/* push	*/
				offset -= hp->line;
		} else if(--depth == 1)		/* pop */
			offset += hp->line;
	}
	*line = ln+offset;
	return 1;
}

/*
 *	compare two encoded file names
 */
static int
hcomp(Hist *hp, short *sp)
{
	uchar *cp;
	int i, j;
	short *s;

	cp = (uchar *)hp->name;
	s = sp;
	if (*s == 0)
		return 0;
	for (i = 1; j = (cp[i]<<8)|cp[i+1]; i += 2) {
		if(j == 0)
			break;
		if(*s == j)
			s++;
		else
			s = sp;
	}
	return *s == 0;
}

/*
 *	Convert a pc to a "file:line {file:line}" string.
 */
int32
fileline(char *str, int n, uvlong dot)
{
	int32 line, top, bot, mid;
	File *f;

	*str = 0;
	if(buildtbls() == 0)
		return 0;
		/* binary search assumes file list is sorted by addr */
	bot = 0;
	top = nfiles;
	for (mid = (bot+top)/2; mid < top; mid = (bot+top)/2) {
		f = &files[mid];
		if(dot < f->addr)
			top = mid;
		else if(mid < nfiles-1 && dot >= (f+1)->addr)
			bot = mid;
		else {
			line = pc2line(dot);
			if(line > 0 && fline(str, n, line, f->hist, 0) >= 0)
				return 1;
			break;
		}
	}
	return 0;
}

/*
 *	Convert a line number within a composite file to relative line
 *	number in a source file.  A composite file is the source
 *	file with included files inserted in line.
 */
static int
fline(char *str, int n, int32 line, Hist *base, Hist **ret)
{
	Hist *start;			/* start of current level */
	Hist *h;			/* current entry */
	int32 delta;			/* sum of size of files this level */
	int k;

	start = base;
	h = base;
	delta = h->line;
	while(h && h->name && line > h->line) {
		if(h->name[1] || h->name[2]) {
			if(h->offset != 0) {	/* #line Directive */
				delta = h->line-h->offset+1;
				start = h;
				base = h++;
			} else {		/* beginning of File */
				if(start == base)
					start = h++;
				else {
					k = fline(str, n, line, start, &h);
					if(k <= 0)
						return k;
				}
			}
		} else {
			if(start == base && ret) {	/* end of recursion level */
				*ret = h;
				return 1;
			} else {			/* end of included file */
				delta += h->line-start->line;
				h++;
				start = base;
			}
		}
	}
	if(!h)
		return -1;
	if(start != base)
		line = line-start->line+1;
	else
		line = line-delta+1;
	if(!h->name)
		strncpy(str, "<eof>", n);
	else {
		k = fileelem(fnames, (uchar*)start->name, str, n);
		if(k+8 < n)
			sprint(str+k, ":%d", line);
	}
/**********Remove comments for complete back-trace of include sequence
 *	if(start != base) {
 *		k = strlen(str);
 *		if(k+2 < n) {
 *			str[k++] = ' ';
 *			str[k++] = '{';
 *		}
 *		k += fileelem(fnames, (uchar*) base->name, str+k, n-k);
 *		if(k+10 < n)
 *			sprint(str+k, ":%ld}", start->line-delta);
 *	}
 ********************/
	return 0;
}

/*
 *	convert an encoded file name to a string.
 */
int
fileelem(Sym **fp, uchar *cp, char *buf, int n)
{
	int i, j;
	char *c, *bp, *end;

	bp = buf;
	end = buf+n-1;
	for(i = 1; j = (cp[i]<<8)|cp[i+1]; i+=2){
		c = fp[j]->name;
		if(bp != buf && bp[-1] != '/' && bp < end)
			*bp++ = '/';
		while(bp < end && *c)
			*bp++ = *c++;
	}
	*bp = 0;
	i =  bp-buf;
	if(i > 1) {
		cleanname(buf);
		i = strlen(buf);
	}
	return i;
}

/*
 *	compare the values of two symbol table entries.
 */
static int
symcomp(const void *a, const void *b)
{
	int i;

	i = (*(Sym**)a)->value - (*(Sym**)b)->value;
	if (i)
		return i;
	return strcmp((*(Sym**)a)->name, (*(Sym**)b)->name);
}

/*
 *	compare the values of the symbols referenced by two text table entries
 */
static int
txtcomp(const void *a, const void *b)
{
	return ((Txtsym*)a)->sym->value - ((Txtsym*)b)->sym->value;
}

/*
 *	compare the values of the symbols referenced by two file table entries
 */
static int
filecomp(const void *a, const void *b)
{
	return ((File*)a)->addr - ((File*)b)->addr;
}

/*
 *	fill an interface Symbol structure from a symbol table entry
 */
static void
fillsym(Sym *sp, Symbol *s)
{
	s->type = sp->type;
	s->value = sp->value;
	s->name = sp->name;
	s->index = 0;
	switch(sp->type) {
	case 'b':
	case 'B':
	case 'D':
	case 'd':
		s->class = CDATA;
		break;
	case 't':
	case 'T':
	case 'l':
	case 'L':
		s->class = CTEXT;
		break;
	case 'a':
		s->class = CAUTO;
		break;
	case 'p':
		s->class = CPARAM;
		break;
	case 'm':
		s->class = CSTAB;
		break;
	default:
		s->class = CNONE;
		break;
	}
	s->handle = 0;
}

/*
 *	find the stack frame, given the pc
 */
uvlong
pc2sp(uvlong pc)
{
	uchar *c, u;
	uvlong currpc, currsp;

	if(spoff == 0)
		return ~0;
	currsp = 0;
	currpc = txtstart - mach->pcquant;

	if(pc<currpc || pc>txtend)
		return ~0;
	for(c = spoff; c < spoffend; c++) {
		if (currpc >= pc)
			return currsp;
		u = *c;
		if (u == 0) {
			currsp += (c[1]<<24)|(c[2]<<16)|(c[3]<<8)|c[4];
			c += 4;
		}
		else if (u < 65)
			currsp += 4*u;
		else if (u < 129)
			currsp -= 4*(u-64);
		else
			currpc += mach->pcquant*(u-129);
		currpc += mach->pcquant;
	}
	return ~0;
}

/*
 *	find the source file line number for a given value of the pc
 */
int32
pc2line(uvlong pc)
{
	uchar *c, u;
	uvlong currpc;
	int32 currline;

	if(pcline == 0)
		return -1;
	currline = 0;
	if (firstinstr != 0)
		currpc = firstinstr-mach->pcquant;
	else
		currpc = txtstart-mach->pcquant;
	if(pc<currpc || pc>txtend)
		return ~0;

	for(c = pcline; c < pclineend && currpc < pc; c++) {
		u = *c;
		if(u == 0) {
			currline += (c[1]<<24)|(c[2]<<16)|(c[3]<<8)|c[4];
			c += 4;
		}
		else if(u < 65)
			currline += u;
		else if(u < 129)
			currline -= (u-64);
		else
			currpc += mach->pcquant*(u-129);
		currpc += mach->pcquant;
	}
	return currline;
}

/*
 *	find the pc associated with a line number
 *	basepc and endpc are text addresses bounding the search.
 *	if endpc == 0, the end of the table is used (i.e., no upper bound).
 *	usually, basepc and endpc contain the first text address in
 *	a file and the first text address in the following file, respectively.
 */
uvlong
line2addr(int32 line, uvlong basepc, uvlong endpc)
{
	uchar *c,  u;
	uvlong currpc, pc;
	int32 currline;
	int32 delta, d;
	int found;

	if(pcline == 0 || line == 0)
		return ~0;

	currline = 0;
	currpc = txtstart-mach->pcquant;
	pc = ~0;
	found = 0;
	delta = HUGEINT;

	for(c = pcline; c < pclineend; c++) {
		if(endpc && currpc >= endpc)	/* end of file of interest */
			break;
		if(currpc >= basepc) {		/* proper file */
			if(currline >= line) {
				d = currline-line;
				found = 1;
			} else
				d = line-currline;
			if(d < delta) {
				delta = d;
				pc = currpc;
			}
		}
		u = *c;
		if(u == 0) {
			currline += (c[1]<<24)|(c[2]<<16)|(c[3]<<8)|c[4];
			c += 4;
		}
		else if(u < 65)
			currline += u;
		else if(u < 129)
			currline -= (u-64);
		else
			currpc += mach->pcquant*(u-129);
		currpc += mach->pcquant;
	}
	if(found)
		return pc;
	return ~0;
}

/*
 *	Print a history stack (debug). if count is 0, prints the whole stack
 */
static void
printhist(char *msg, Hist *hp, int count)
{
	int i;
	uchar *cp;
	char buf[128];

	i = 0;
	while(hp->name) {
		if(count && ++i > count)
			break;
		print("%s Line: %x (%d)  Offset: %x (%d)  Name: ", msg,
			hp->line, hp->line, hp->offset, hp->offset);
		for(cp = (uchar *)hp->name+1; (*cp<<8)|cp[1]; cp += 2) {
			if (cp != (uchar *)hp->name+1)
				print("/");
			print("%x", (*cp<<8)|cp[1]);
		}
		fileelem(fnames, (uchar *) hp->name, buf, sizeof(buf));
		print(" (%s)\n", buf);
		hp++;
	}
}

#ifdef DEBUG
/*
 *	print the history stack for a file. (debug only)
 *	if (name == 0) => print all history stacks.
 */
void
dumphist(char *name)
{
	int i;
	File *f;
	short *fname;

	if(buildtbls() == 0)
		return;
	if(name)
		fname = encfname(name);
	for(i = 0, f = files; i < nfiles; i++, f++)
		if(fname == 0 || hcomp(f->hist, fname))
			printhist("> ", f->hist, f->n);

	if(fname)
		free(fname);
}
#endif
