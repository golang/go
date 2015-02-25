// Derived from Inferno utils/6l/l.h and related files.
// http://code.google.com/p/inferno-os/source/browse/utils/6l/l.h
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
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

typedef	struct	LSym	LSym;
typedef	struct	Reloc	Reloc;
typedef	struct	Auto	Auto;
typedef	struct	Link	Link;
typedef	struct	LinkArch	LinkArch;
typedef	struct	Library	Library;

typedef	struct	Pcln	Pcln;
typedef	struct	Pcdata	Pcdata;
typedef	struct	Pciter	Pciter;

struct	Reloc
{
	int32	off;
	uchar	siz;
	uchar	done;
	int32	type;
	int32	variant; // RV_*: variant on computed value
	int64	add;
	int64	xadd;
	LSym*	sym;
	LSym*	xsym;
};

struct	LSym
{
	char*	name;
	char*	extname;	// name used in external object files
	short	type;
	short	version;
	uchar	dupok;
	uchar	cfunc;
	uchar	external;
	uchar	nosplit;
	uchar	reachable;
	uchar	cgoexport;
	uchar	special;
	uchar	stkcheck;
	uchar	hide;
	uchar	leaf;	// arm only
	uchar	localentry;	// ppc64: instrs between global & local entry
	uchar	onlist;	// on the textp or datap lists
	int32	dynid;
	int32	sig;
	int32	plt;
	int32	got;
	int32	align;	// if non-zero, required alignment in bytes
	int32	elfsym;
	int32	args;	// size of stack frame incoming arguments area
	int32	locals;	// size of stack frame locals area (arm only?)
	vlong	value;
	vlong	size;
	LSym*	hash;	// in hash table
	LSym*	allsym;	// in all symbol list
	LSym*	next;	// in text or data list
	LSym*	sub;	// in SSUB list
	LSym*	outer;	// container of sub
	LSym*	gotype;
	LSym*	reachparent;
	LSym*	queue;
	char*	file;
	char*	dynimplib;
	char*	dynimpvers;
	void*	sect;
	
	// STEXT
	Auto*	autom;
	Pcln*	pcln;

	// SDATA, SBSS
	uchar*	p;
	int	np;
	int32	maxp;
	Reloc*	r;
	int32	nr;
	int32	maxr;
};

// LSym.type
enum
{
	Sxxx,

	/* order here is order in output file */
	/* readonly, executable */
	STEXT,
	SELFRXSECT,
	
	/* readonly, non-executable */
	STYPE,
	SSTRING,
	SGOSTRING,
	SGOFUNC,
	SRODATA,
	SFUNCTAB,
	STYPELINK,
	SSYMTAB, // TODO: move to unmapped section
	SPCLNTAB,
	SELFROSECT,
	
	/* writable, non-executable */
	SMACHOPLT,
	SELFSECT,
	SMACHO,	/* Mach-O __nl_symbol_ptr */
	SMACHOGOT,
	SWINDOWS,
	SELFGOT,	/* also .toc in ppc64 ABI */
	SNOPTRDATA,
	SINITARR,
	SDATA,
	SBSS,
	SNOPTRBSS,
	STLSBSS,

	/* not mapped */
	SXREF,
	SMACHOSYMSTR,
	SMACHOSYMTAB,
	SMACHOINDIRECTPLT,
	SMACHOINDIRECTGOT,
	SFILE,
	SFILEPATH,
	SCONST,
	SDYNIMPORT,
	SHOSTOBJ,

	SSUB = 1<<8,	/* sub-symbol, linked from parent via ->sub list */
	SMASK = SSUB - 1,
	SHIDDEN = 1<<9, // hidden or local symbol
};

// Reloc.type
enum
{
	R_ADDR = 1,
	R_ADDRPOWER, // relocation for loading 31-bit address using addis and addi/ld/st for Power
	R_SIZE,
	R_CALL, // relocation for direct PC-relative call
	R_CALLARM, // relocation for ARM direct call
	R_CALLIND, // marker for indirect call (no actual relocating necessary)
	R_CALLPOWER, // relocation for Power direct call
	R_CONST,
	R_PCREL,
	R_TLS,
	R_TLS_LE, // TLS local exec offset from TLS segment register
	R_TLS_IE, // TLS initial exec offset from TLS base pointer
	R_GOTOFF,
	R_PLT0,
	R_PLT1,
	R_PLT2,
	R_USEFIELD,
	R_POWER_TOC,		// ELF R_PPC64_TOC16*
};

// Reloc.variant
enum
{
	RV_NONE,		// identity variant
	RV_POWER_LO,		// x & 0xFFFF
	RV_POWER_HI,		// x >> 16
	RV_POWER_HA,		// (x + 0x8000) >> 16
	RV_POWER_DS,		// x & 0xFFFC, check x&0x3 == 0

	RV_CHECK_OVERFLOW = 1<<8,	// check overflow flag
	RV_TYPE_MASK = (RV_CHECK_OVERFLOW - 1),
};

// Auto.name
enum
{
	A_AUTO = 1,
	A_PARAM,
};

struct	Auto
{
	LSym*	asym;
	Auto*	link;
	int32	aoffset;
	int16	name;
	LSym*	gotype;
};

enum
{
	LINKHASH = 100003,
};

struct	Library
{
	char *objref;	// object where we found the reference
	char *srcref;	// src file where we found the reference
	char *file;	// object file
	char *pkg;	// import path
};

struct Pcdata
{
	uchar *p;
	int n;
	int m;
};

struct Pcln
{
	Pcdata pcsp;
	Pcdata pcfile;
	Pcdata pcline;
	Pcdata *pcdata;
	int npcdata;
	LSym **funcdata;
	int64 *funcdataoff;
	int nfuncdata;
	
	LSym **file;
	int nfile;
	int mfile;

	LSym *lastfile;
	int lastindex;
};

// Pcdata iterator.
//	for(pciterinit(ctxt, &it, &pcd); !it.done; pciternext(&it)) { it.value holds in [it.pc, it.nextpc) }
struct Pciter
{
	Pcdata d;
	uchar *p;
	uint32 pc;
	uint32 nextpc;
	uint32 pcscale;
	int32 value;
	int start;
	int done;
};

void	pciterinit(Link*, Pciter*, Pcdata*);
void	pciternext(Pciter*);

// symbol version, incremented each time a file is loaded.
// version==1 is reserved for savehist.
enum
{
	HistVersion = 1,
};

// Link holds the context for writing object code from a compiler
// to be linker input or for reading that input into the linker.
struct	Link
{
	int32	thechar; // '5' (arm), '6' (amd64), etc.
	char*	thestring; // full name of architecture ("arm", "amd64", ..)
	int32	goarm; // for arm only, GOARM setting
	int	headtype;

	LinkArch*	arch;
	int32	debugasm;	// -S flag in compiler
	int32	debugvlog;	// -v flag in linker
	Biobuf*	bso;	// for -v flag
	int32	windows;
	char*	goroot;

	// hash table of all symbols
	LSym*	hash[LINKHASH];
	LSym*	allsym;
	int32	nsymbol;
	
	// code generation
	LSym*	tlsg;

	// for reading input files (during linker)
	char**	libdir;
	int32	nlibdir;
	int32	maxlibdir;
	Library*	library;
	int	libraryp;
	int	nlibrary;
	int	tlsoffset;
	void	(*diag)(char*, ...);
	LSym*	cursym;
	int	version;
	LSym*	textp;
	LSym*	etextp;
	int32	nhistfile;
	LSym*	filesyms;
};

enum {
	LittleEndian = 0x04030201,
	BigEndian = 0x01020304,
};

// LinkArch is the definition of a single architecture.
struct LinkArch
{
	char*	name; // "arm", "amd64", and so on
	int	thechar;	// '5', '6', and so on
	int32	endian; // LittleEndian or BigEndian

	int	minlc;
	int	ptrsize;
	int	regsize;
};

/* executable header types */
enum {
	Hunknown = 0,
	Hdarwin,
	Hdragonfly,
	Helf,
	Hfreebsd,
	Hlinux,
	Hnacl,
	Hnetbsd,
	Hopenbsd,
	Hplan9,
	Hsolaris,
	Hwindows,
};

enum
{
	LinkAuto = 0,
	LinkInternal,
	LinkExternal,
};

extern	uchar	fnuxi8[8];
extern	uchar	fnuxi4[4];
extern	uchar	inuxi1[1];
extern	uchar	inuxi2[2];
extern	uchar	inuxi4[4];
extern	uchar	inuxi8[8];

vlong	addaddr(Link *ctxt, LSym *s, LSym *t);
vlong	addaddrplus(Link *ctxt, LSym *s, LSym *t, vlong add);
vlong	addaddrplus4(Link *ctxt, LSym *s, LSym *t, vlong add);
void	addlib(Link *ctxt, char *src, char *obj, char *pathname);
void	addlibpath(Link *ctxt, char *srcref, char *objref, char *file, char *pkg);
vlong	addpcrelplus(Link *ctxt, LSym *s, LSym *t, vlong add);
Reloc*	addrel(LSym *s);
vlong	addsize(Link *ctxt, LSym *s, LSym *t);
vlong	adduint16(Link *ctxt, LSym *s, uint16 v);
vlong	adduint32(Link *ctxt, LSym *s, uint32 v);
vlong	adduint64(Link *ctxt, LSym *s, uint64 v);
vlong	adduint8(Link *ctxt, LSym *s, uint8 v);
vlong	adduintxx(Link *ctxt, LSym *s, uint64 v, int wid);
vlong	atolwhex(char *s);
void*	emallocz(long n);
void*	erealloc(void *p, long n);
char*	estrdup(char *p);
char*	expandpkg(char *t0, char *pkg);
int	find1(int32 l, int c);
char*	headstr(int v);
int	headtype(char *name);
void	ldobjfile(Link *ctxt, Biobuf *f, char *pkg, int64 len, char *pn);
LSym*	linklookup(Link *ctxt, char *name, int v);
Link*	linknew(LinkArch *arch);
LSym*	linknewsym(Link *ctxt, char *symb, int v);
LSym*	linkrlookup(Link *ctxt, char *name, int v);
void	nuxiinit(LinkArch *arch);
void	pciterinit(Link *ctxt, Pciter *it, Pcdata *d);
void	pciternext(Pciter *it);
vlong	setaddr(Link *ctxt, LSym *s, vlong off, LSym *t);
vlong	setaddrplus(Link *ctxt, LSym *s, vlong off, LSym *t, vlong add);
vlong	setuint32(Link *ctxt, LSym *s, vlong r, uint32 v);
vlong	setuint8(Link *ctxt, LSym *s, vlong r, uint8 v);
vlong	setuintxx(Link *ctxt, LSym *s, vlong off, uint64 v, vlong wid);
void	symgrow(Link *ctxt, LSym *s, vlong lsiz);

extern	LinkArch	link386;
extern	LinkArch	linkamd64;
extern	LinkArch	linkamd64p32;
extern	LinkArch	linkarm;
extern	LinkArch	linkppc64;
extern	LinkArch	linkppc64le;
