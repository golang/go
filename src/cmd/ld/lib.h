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

// Where symbol table data gets mapped into memory.
#define SYMDATVA 0x99LL<<24

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

struct Section
{
	uchar	rwx;
	char	*name;
	uvlong	vaddr;
	uvlong	len;
	Section	*next;	// in segment list
	Segment	*seg;
};

extern	char	symname[];
extern	char	*libdir[];
extern	int	nlibdir;
extern	int	cout;

EXTERN	char*	INITENTRY;
EXTERN	char	thechar;
EXTERN	char*	thestring;
EXTERN	Library*	library;
EXTERN	int	libraryp;
EXTERN	int	nlibrary;
EXTERN	Sym*	hash[NHASH];
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

EXTERN	Segment	segtext;
EXTERN	Segment	segdata;
EXTERN	Segment	segrodata;	// NaCl only
EXTERN	Segment	segsym;

void	addlib(char *src, char *obj);
void	addlibpath(char *srcref, char *objref, char *file, char *pkg);
Section*	addsection(Segment*, char*, int);
void	copyhistfrog(char *buf, int nbuf);
void	addhist(int32 line, int type);
void	asmlc(void);
void	histtoauto(void);
void	collapsefrog(Sym *s);
Sym*	lookup(char *symb, int v);
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
void	Lflag(char *arg);
void	usage(void);
void	ldobj1(Biobuf *f, char*, int64 len, char *pn);
void	ldobj(Biobuf*, char*, int64, char*, int);
void	ldpkg(Biobuf*, char*, int64, char*, int);
void	mark(Sym *s);
void	mkfwd(void);
char*	expandpkg(char*, char*);
void	deadcode(void);
void	ewrite(int, void*, int);

int	pathchar(void);
void*	mal(uint32);
void	mywhatsys(void);

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
