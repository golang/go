// Derived from Inferno utils/6l/l.h
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

enum
{
	Sxxx,

	/* order here is order in output file */
	STEXT,
	STYPE,
	SSTRING,
	SGOSTRING,
	SRODATA,
	STYPELINK,
	SSYMTAB,
	SPCLNTAB,
	SELFROSECT,
	SMACHOPLT,
	SELFSECT,
	SMACHO,	/* Mach-O __nl_symbol_ptr */
	SMACHOGOT,
	SNOPTRDATA,
	SDATARELRO,
	SDATA,
	SWINDOWS,
	SBSS,
	SNOPTRBSS,
	STLSBSS,

	SXREF,
	SMACHOSYMSTR,
	SMACHOSYMTAB,
	SMACHOINDIRECTPLT,
	SMACHOINDIRECTGOT,
	SFILE,
	SCONST,
	SDYNIMPORT,
	SHOSTOBJ,

	SSUB = 1<<8,	/* sub-symbol, linked from parent via ->sub list */
	SMASK = SSUB - 1,
	SHIDDEN = 1<<9, // hidden or local symbol

	NHASH = 100003,
};

enum
{
	// This value is known to the garbage collector and should be kept in
	// sync with runtime/pkg/runtime.h
	ArgsSizeUnknown = 0x80000000
};

typedef struct Library Library;
struct Library
{
	char *objref;	// object where we found the reference
	char *srcref;	// src file where we found the reference
	char *file;		// object file
	char *pkg;	// import path
};

// Terrible but standard terminology.
// A segment describes a block of file to load into memory.
// A section further describes the pieces of that block for
// use in debuggers and such.

typedef struct Segment Segment;
typedef struct Section Section;

struct Segment
{
	uchar	rwx;		// permission as usual unix bits (5 = r-x etc)
	uvlong	vaddr;	// virtual address
	uvlong	len;		// length in memory
	uvlong	fileoff;	// file offset
	uvlong	filelen;	// length on disk
	Section*	sect;
};

#pragma incomplete struct Elf64_Shdr

struct Section
{
	uchar	rwx;
	int16	extnum;
	int32	align;
	char	*name;
	uvlong	vaddr;
	uvlong	len;
	Section	*next;	// in segment list
	Segment	*seg;
	struct Elf64_Shdr *elfsect;
	uvlong	reloff;
	uvlong	rellen;
};

extern	char	symname[];
extern	char	**libdir;
extern	int	nlibdir;

EXTERN	char*	INITENTRY;
EXTERN	char*	thestring;
EXTERN	Library*	library;
EXTERN	int	libraryp;
EXTERN	int	nlibrary;
EXTERN	Sym*	hash[NHASH];
EXTERN	Sym*	allsym;
EXTERN	Sym*	histfrog[MAXHIST];
EXTERN	uchar	fnuxi8[8];
EXTERN	uchar	fnuxi4[4];
EXTERN	int	histfrogp;
EXTERN	int	histgen;
EXTERN	uchar	inuxi1[1];
EXTERN	uchar	inuxi2[2];
EXTERN	uchar	inuxi4[4];
EXTERN	uchar	inuxi8[8];
EXTERN	char*	outfile;
EXTERN	int32	nsymbol;
EXTERN	char*	thestring;
EXTERN	int	ndynexp;
EXTERN	Sym**	dynexp;
EXTERN	int	nldflag;
EXTERN	char**	ldflag;
EXTERN	int	havedynamic;
EXTERN	int	iscgo;
EXTERN	int	elfglobalsymndx;
EXTERN	int	flag_race;
EXTERN	int flag_shared;
EXTERN	char*	tracksym;
EXTERN	char*	interpreter;
EXTERN	char*	tmpdir;
EXTERN	char*	extld;
EXTERN	char*	extldflags;

enum
{
	LinkAuto = 0,
	LinkInternal,
	LinkExternal,
};
EXTERN	int	linkmode;

// for dynexport field of Sym
enum
{
	CgoExportDynamic = 1<<0,
	CgoExportStatic = 1<<1,
};

EXTERN	Segment	segtext;
EXTERN	Segment	segdata;
EXTERN	Segment	segdwarf;

void	setlinkmode(char*);
void	addlib(char *src, char *obj);
void	addlibpath(char *srcref, char *objref, char *file, char *pkg);
Section*	addsection(Segment*, char*, int);
void	copyhistfrog(char *buf, int nbuf);
void	addhist(int32 line, int type);
void	asmlc(void);
void	histtoauto(void);
void	collapsefrog(Sym *s);
Sym*	newsym(char *symb, int v);
Sym*	lookup(char *symb, int v);
Sym*	rlookup(char *symb, int v);
void	nuxiinit(void);
int	find1(int32 l, int c);
int	find2(int32 l, int c);
int32	ieeedtof(Ieee *e);
double	ieeedtod(Ieee *e);
void	undefsym(Sym *s);
void	zerosig(char *sp);
void	readundefs(char *f, int t);
int32	Bget4(Biobuf *f);
void	loadlib(void);
void	errorexit(void);
void	mangle(char*);
void	objfile(char *file, char *pkg);
void	libinit(void);
void	pclntab(void);
void	symtab(void);
void	Lflag(char *arg);
void	usage(void);
void	adddynrel(Sym*, Reloc*);
void	adddynrela(Sym*, Sym*, Reloc*);
Sym*	lookuprel(void);
void	ldobj1(Biobuf *f, char*, int64 len, char *pn);
void	ldobj(Biobuf*, char*, int64, char*, char*, int);
void	ldelf(Biobuf*, char*, int64, char*);
void	ldmacho(Biobuf*, char*, int64, char*);
void	ldpe(Biobuf*, char*, int64, char*);
void	ldpkg(Biobuf*, char*, int64, char*, int);
void	mark(Sym *s);
void	mkfwd(void);
char*	expandpkg(char*, char*);
void	deadcode(void);
Reloc*	addrel(Sym*);
void	codeblk(int32, int32);
void	datblk(int32, int32);
void	reloc(void);
void	relocsym(Sym*);
void	savedata(Sym*, Prog*, char*);
void	symgrow(Sym*, int32);
void	addstrdata(char*, char*);
vlong	addstring(Sym*, char*);
vlong	adduint8(Sym*, uint8);
vlong	adduint16(Sym*, uint16);
vlong	adduint32(Sym*, uint32);
vlong	adduint64(Sym*, uint64);
vlong	adduintxx(Sym*, uint64, int);
vlong	addaddr(Sym*, Sym*);
vlong	addaddrplus(Sym*, Sym*, vlong);
vlong	addpcrelplus(Sym*, Sym*, vlong);
vlong	addsize(Sym*, Sym*);
vlong	setaddrplus(Sym*, vlong, Sym*, vlong);
vlong	setaddr(Sym*, vlong, Sym*);
void	setuint8(Sym*, vlong, uint8);
void	setuint16(Sym*, vlong, uint16);
void	setuint32(Sym*, vlong, uint32);
void	setuint64(Sym*, vlong, uint64);
void	asmsym(void);
void	asmelfsym(void);
void	asmplan9sym(void);
void	putelfsectionsym(Sym*, int);
void	putelfsymshndx(vlong, int);
void	strnput(char*, int);
void	dodata(void);
void	dosymtype(void);
void	address(void);
void	textaddress(void);
void	genasmsym(void (*put)(Sym*, char*, int, vlong, vlong, int, Sym*));
vlong	datoff(vlong);
void	adddynlib(char*);
int	archreloc(Reloc*, Sym*, vlong*);
void	adddynsym(Sym*);
void	addexport(void);
void	dostkcheck(void);
void	undef(void);
void	doweak(void);
void	setpersrc(Sym*);
void	doversion(void);
void	usage(void);
void	setinterp(char*);
Sym*	listsort(Sym*, int(*cmp)(Sym*, Sym*), int);
int	valuecmp(Sym*, Sym*);
void	hostobjs(void);
void	hostlink(void);
char*	estrdup(char*);
void*	erealloc(void*, long);

int	pathchar(void);
void*	mal(uint32);
void	unmal(void*, uint32);
void	mywhatsys(void);
int	rbyoff(const void*, const void*);

uint16	le16(uchar*);
uint32	le32(uchar*);
uint64	le64(uchar*);
uint16	be16(uchar*);
uint32	be32(uchar*);
uint64	be64(uchar*);

typedef struct Endian Endian;
struct Endian
{
	uint16	(*e16)(uchar*);
	uint32	(*e32)(uchar*);
	uint64	(*e64)(uchar*);
};

extern Endian be, le;

/* set by call to mywhatsys() */
extern	char*	goroot;
extern	char*	goarch;
extern	char*	goos;

/* whence for ldpkg */
enum {
	FileObj = 0,
	ArchiveObj,
	Pkgdef
};

/* executable header types */
enum {
	Hgarbunix = 0,	// garbage unix
	Hnoheader,	// no header
	Hunixcoff,	// unix coff
	Hrisc,		// aif for risc os
	Hplan9x32,	// plan 9 32-bit format
	Hplan9x64,	// plan 9 64-bit format
	Hmsdoscom,	// MS-DOS .COM
	Hnetbsd,	// NetBSD
	Hmsdosexe,	// fake MS-DOS .EXE
	Hixp1200,	// IXP1200 (raw)
	Helf,		// ELF32
	Hipaq,		// ipaq
	Hdarwin,	// Apple Mach-O
	Hlinux,		// Linux ELF
	Hfreebsd,	// FreeBSD ELF
	Hwindows,	// MS Windows PE
	Hopenbsd,	// OpenBSD ELF
};

typedef struct Header Header;
struct Header {
	char *name;
	int val;
};

EXTERN	char*	headstring;
extern	Header	headers[];

int	headtype(char*);
char*	headstr(int);
void	setheadtype(char*);

int	Yconv(Fmt*);

#pragma	varargck	type	"O"	int
#pragma	varargck	type	"Y"	Sym*

// buffered output

EXTERN	Biobuf	bso;

EXTERN struct
{
	char	cbuf[MAXIO];	/* output buffer */
} buf;

EXTERN	int	cbc;
EXTERN	char*	cbp;
EXTERN	char*	cbpmax;

#define	cput(c)\
	{ *cbp++ = c;\
	if(--cbc <= 0)\
		cflush(); }

void	cflush(void);
vlong	cpos(void);
void	cseek(vlong);
void	cwrite(void*, int);
void	importcycles(void);
int	Zconv(Fmt*);

uint8	decodetype_kind(Sym*);
vlong	decodetype_size(Sym*);
Sym*	decodetype_gc(Sym*);
Sym*	decodetype_arrayelem(Sym*);
vlong	decodetype_arraylen(Sym*);
Sym*	decodetype_ptrelem(Sym*);
Sym*	decodetype_mapkey(Sym*);
Sym*	decodetype_mapvalue(Sym*);
Sym*	decodetype_chanelem(Sym*);
int	decodetype_funcdotdotdot(Sym*);
int	decodetype_funcincount(Sym*);
int	decodetype_funcoutcount(Sym*);
Sym*	decodetype_funcintype(Sym*, int);
Sym*	decodetype_funcouttype(Sym*, int);
int	decodetype_structfieldcount(Sym*);
char*	decodetype_structfieldname(Sym*, int);
Sym*	decodetype_structfieldtype(Sym*, int);
vlong	decodetype_structfieldoffs(Sym*, int);
vlong	decodetype_ifacemethodcount(Sym*);
