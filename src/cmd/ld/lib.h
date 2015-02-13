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

#ifndef	EXTERN
#define	EXTERN	extern
#endif

typedef struct Arch Arch;
struct Arch {
	int	thechar;
	int	ptrsize;
	int	intsize;
	int	regsize;
	int	funcalign;
	int	maxalign;
	int	minlc;
	int	dwarfregsp;
	
	char *linuxdynld;
	char *freebsddynld;
	char *netbsddynld;
	char *openbsddynld;
	char *dragonflydynld;
	char *solarisdynld;
	
	void	(*adddynlib)(char*);
	void	(*adddynrel)(LSym*, Reloc*);
	void	(*adddynsym)(Link*, LSym*);
	void	(*archinit)(void);
	int	(*archreloc)(Reloc*, LSym*, vlong*);
	vlong	(*archrelocvariant)(Reloc*, LSym*, vlong);
	void	(*asmb)(void);
	int	(*elfreloc1)(Reloc*, vlong);
	void	(*elfsetupplt)(void);
	void	(*gentext)(void);
	void	(*listinit)(void);
	int	(*machoreloc1)(Reloc*, vlong);
	
	void	(*lput)(uint32);
	void	(*wput)(uint16);
	void	(*vput)(uint64);
};

vlong rnd(vlong, vlong);

EXTERN	Arch	thearch;
EXTERN	LSym*	datap;
EXTERN	int	debug[128];
EXTERN	char	literal[32];
EXTERN	int32	lcsize;
EXTERN	char*	rpath;
EXTERN	int32	spsize;
EXTERN	LSym*	symlist;
EXTERN	int32	symsize;

// Terrible but standard terminology.
// A segment describes a block of file to load into memory.
// A section further describes the pieces of that block for
// use in debuggers and such.

enum {
	MAXIO		= 8192,
	MINFUNC		= 16,	// minimum size for a function
};

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

EXTERN	char*	INITENTRY;
EXTERN	char*	thestring;
EXTERN	LinkArch*	thelinkarch;
EXTERN	char*	outfile;
EXTERN	int	ndynexp;
EXTERN	LSym**	dynexp;
EXTERN	int	nldflag;
EXTERN	char**	ldflag;
EXTERN	int	havedynamic;
EXTERN	int	funcalign;
EXTERN	int	iscgo;
EXTERN	int	elfglobalsymndx;
extern	int	nelfsym;
EXTERN	char*	flag_installsuffix;
EXTERN	int	flag_race;
EXTERN	int flag_shared;
EXTERN	char*	tracksym;
EXTERN	char*	interpreter;
EXTERN	char*	tmpdir;
EXTERN	char*	extld;
EXTERN	char*	extldflags;
EXTERN	int	debug_s; // backup old value of debug['s']
EXTERN	Link*	ctxt;
EXTERN	int32	HEADR;
EXTERN	int32	HEADTYPE;
EXTERN	int32	INITRND;
EXTERN	int64	INITTEXT;
EXTERN	int64	INITDAT;
EXTERN	char*	INITENTRY;		/* entry point */
EXTERN	char*	noname;
EXTERN	char*	paramspace;
EXTERN	int	nerrors;

EXTERN	int	linkmode;
EXTERN	int64	liveness;

// for dynexport field of LSym
enum
{
	CgoExportDynamic = 1<<0,
	CgoExportStatic = 1<<1,
};

EXTERN	Segment	segtext;
EXTERN	Segment	segrodata;
EXTERN	Segment	segdata;
EXTERN	Segment	segdwarf;

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

typedef struct Header Header;
struct Header {
	char *name;
	int val;
};

EXTERN	char*	headstring;
extern	Header	headers[];

#pragma	varargck	type	"Y"	LSym*
#pragma	varargck	type	"Z"	char*
#pragma	varargck	type	"i"	char*

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

void	Lflag(char *arg);
int	Yconv(Fmt *fp);
int	Zconv(Fmt *fp);
void	addexport(void);
void	address(void);
Section*addsection(Segment *seg, char *name, int rwx);
void	addstrdata(char *name, char *value);
vlong	addstring(LSym *s, char *str);
void	asmelfsym(void);
void	asmplan9sym(void);
uint16	be16(uchar *b);
uint32	be32(uchar *b);
uint64	be64(uchar *b);
void	callgraph(void);
void	checkgo(void);
void	cflush(void);
void	codeblk(int64 addr, int64 size);
vlong	cpos(void);
void	cseek(vlong p);
void	cwrite(void *buf, int n);
void	datblk(int64 addr, int64 size);
int	datcmp(LSym *s1, LSym *s2);
vlong	datoff(vlong addr);
void	deadcode(void);
LSym*	decodetype_arrayelem(LSym *s);
vlong	decodetype_arraylen(LSym *s);
LSym*	decodetype_chanelem(LSym *s);
int	decodetype_funcdotdotdot(LSym *s);
int	decodetype_funcincount(LSym *s);
LSym*	decodetype_funcintype(LSym *s, int i);
int	decodetype_funcoutcount(LSym *s);
LSym*	decodetype_funcouttype(LSym *s, int i);
LSym*	decodetype_gcprog(LSym *s);
uint8*	decodetype_gcmask(LSym *s);
vlong	decodetype_ifacemethodcount(LSym *s);
uint8	decodetype_kind(LSym *s);
uint8	decodetype_noptr(LSym *s);
uint8	decodetype_usegcprog(LSym *s);
LSym*	decodetype_mapkey(LSym *s);
LSym*	decodetype_mapvalue(LSym *s);
LSym*	decodetype_ptrelem(LSym *s);
vlong	decodetype_size(LSym *s);
int	decodetype_structfieldcount(LSym *s);
char*	decodetype_structfieldname(LSym *s, int i);
vlong	decodetype_structfieldoffs(LSym *s, int i);
LSym*	decodetype_structfieldtype(LSym *s, int i);
void	dodata(void);
void	dostkcheck(void);
void	dostkoff(void);
void	dosymtype(void);
void	doversion(void);
void	doweak(void);
void	dynreloc(void);
void	dynrelocsym(LSym *s);
vlong	entryvalue(void);
void	errorexit(void);
void	follow(void);
void	genasmsym(void (*put)(LSym*, char*, int, vlong, vlong, int, LSym*));
void	gentext(void);
void	growdatsize(vlong *datsizep, LSym *s);
char*	headstr(int v);
int	headtype(char *name);
void	hostlink(void);
void	hostobjs(void);
int	iconv(Fmt *fp);
void	importcycles(void);
void	linkarchinit(void);
void	ldelf(Biobuf *f, char *pkg, int64 len, char *pn);
void	ldhostobj(void (*ld)(Biobuf*, char*, int64, char*), Biobuf *f, char *pkg, int64 len, char *pn, char *file);
void	ldmacho(Biobuf *f, char *pkg, int64 len, char *pn);
void	ldobj(Biobuf *f, char *pkg, int64 len, char *pn, char *file, int whence);
void	ldpe(Biobuf *f, char *pkg, int64 len, char *pn);
void	ldpkg(Biobuf *f, char *pkg, int64 len, char *filename, int whence);
uint16	le16(uchar *b);
uint32	le32(uchar *b);
uint64	le64(uchar *b);
void	libinit(void);
LSym*	listsort(LSym *l, int (*cmp)(LSym*, LSym*), int off);
void	loadinternal(char *name);
void	loadlib(void);
void	lputb(uint32 l);
void	lputl(uint32 l);
void*	mal(uint32 n);
void	mark(LSym *s);
void	mywhatsys(void);
struct ar_hdr;
void	objfile(char *file, char *pkg);
void	patch(void);
int	pathchar(void);
void	pcln(void);
void	pclntab(void);
void	findfunctab(void);
void	putelfsectionsym(LSym* s, int shndx);
void	putelfsymshndx(vlong sympos, int shndx);
void	putsymb(LSym *s, char *name, int t, vlong v, vlong size, int ver, LSym *typ);
int	rbyoff(const void *va, const void *vb);
void	reloc(void);
void	relocsym(LSym *s);
void	setheadtype(char *s);
void	setinterp(char *s);
void	setlinkmode(char *arg);
void	span(void);
void	strnput(char *s, int n);
vlong	symaddr(LSym *s);
void	symtab(void);
void	textaddress(void);
void	undef(void);
void	unmal(void *v, uint32 n);
void	usage(void);
void	vputb(uint64 v);
int	valuecmp(LSym *a, LSym *b);
void	vputl(uint64 v);
void	wputb(ushort w);
void	wputl(ushort w);
void	xdefine(char *p, int t, vlong v);
void	zerosig(char *sp);
void	archinit(void);
void	diag(char *fmt, ...);

void	ldmain(int, char**);

#pragma	varargck	argpos	diag	1

#define	SYMDEF	"__.GOSYMDEF"