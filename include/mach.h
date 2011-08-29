// Inferno libmach/a.out.h and libmach/mach.h
// http://code.google.com/p/inferno-os/source/browse/utils/libmach/a.out.h
// http://code.google.com/p/inferno-os/source/browse/utils/libmach/mach.h
//
// 	Copyright © 1994-1999 Lucent Technologies Inc.
// 	Power PC support Copyright © 1995-2004 C H Forsyth (forsyth@terzarima.net).
// 	Portions Copyright © 1997-1999 Vita Nuova Limited.
// 	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com).
// 	Revisions Copyright © 2000-2004 Lucent Technologies Inc. and others.
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

/*
 *	Architecture-dependent application data
 */

typedef	struct	Exec	Exec;
struct	Exec
{
	int32	magic;		/* magic number */
	int32	text;	 	/* size of text segment */
	int32	data;	 	/* size of initialized data */
	int32	bss;	  	/* size of uninitialized data */
	int32	syms;	 	/* size of symbol table */
	int32	entry;	 	/* entry point */
	int32	spsz;		/* size of pc/sp offset table */
	int32	pcsz;		/* size of pc/line number table */
};

#define HDR_MAGIC	0x00008000		/* header expansion */

#define	_MAGIC(f, b)	((f)|((((4*(b))+0)*(b))+7))
#define	A_MAGIC		_MAGIC(0, 8)		/* 68020 */
#define	I_MAGIC		_MAGIC(0, 11)		/* intel 386 */
#define	J_MAGIC		_MAGIC(0, 12)		/* intel 960 (retired) */
#define	K_MAGIC		_MAGIC(0, 13)		/* sparc */
#define	V_MAGIC		_MAGIC(0, 16)		/* mips 3000 BE */
#define	X_MAGIC		_MAGIC(0, 17)		/* att dsp 3210 (retired) */
#define	M_MAGIC		_MAGIC(0, 18)		/* mips 4000 BE */
#define	D_MAGIC		_MAGIC(0, 19)		/* amd 29000 (retired) */
#define	E_MAGIC		_MAGIC(0, 20)		/* arm */
#define	Q_MAGIC		_MAGIC(0, 21)		/* powerpc */
#define	N_MAGIC		_MAGIC(0, 22)		/* mips 4000 LE */
#define	L_MAGIC		_MAGIC(0, 23)		/* dec alpha */
#define	P_MAGIC		_MAGIC(0, 24)		/* mips 3000 LE */
#define	U_MAGIC		_MAGIC(0, 25)		/* sparc64 */
#define	S_MAGIC		_MAGIC(HDR_MAGIC, 26)	/* amd64 */
#define	T_MAGIC		_MAGIC(HDR_MAGIC, 27)	/* powerpc64 */

#define	MIN_MAGIC	8
#define	MAX_MAGIC	27			/* <= 90 */

#define	DYN_MAGIC	0x80000000		/* dlm */

typedef	struct	Sym	Sym;
struct	Sym
{
	vlong	value;
	uint	sig;
	char	type;
	char	*name;
	vlong	gotype;
	int	sequence;	// order in file
};


/*
 *	Supported architectures:
 *		mips,
 *		68020,
 *		i386,
 *		amd64,
 *		sparc,
 *		sparc64,
 *		mips2 (R4000)
 *		arm
 *		powerpc,
 *		powerpc64
 *		alpha
 */
enum
{
	MMIPS,			/* machine types */
	MSPARC,
	M68020,
	MI386,
	MI960,			/* retired */
	M3210,			/* retired */
	MMIPS2,
	NMIPS2,
	M29000,			/* retired */
	MARM,
	MPOWER,
	MALPHA,
	NMIPS,
	MSPARC64,
	MAMD64,
	MPOWER64,
				/* types of executables */
	FNONE = 0,		/* unidentified */
	FMIPS,			/* v.out */
	FMIPSB,			/* mips bootable */
	FSPARC,			/* k.out */
	FSPARCB,		/* Sparc bootable */
	F68020,			/* 2.out */
	F68020B,		/* 68020 bootable */
	FNEXTB,			/* Next bootable */
	FI386,			/* 8.out */
	FI386B,			/* I386 bootable */
	FI960,			/* retired */
	FI960B,			/* retired */
	F3210,			/* retired */
	FMIPS2BE,		/* 4.out */
	F29000,			/* retired */
	FARM,			/* 5.out */
	FARMB,			/* ARM bootable */
	FPOWER,			/* q.out */
	FPOWERB,		/* power pc bootable */
	FMIPS2LE,		/* 0.out */
	FALPHA,			/* 7.out */
	FALPHAB,		/* DEC Alpha bootable */
	FMIPSLE,		/* 3k little endian */
	FSPARC64,		/* u.out */
	FAMD64,			/* 6.out */
	FAMD64B,		/* 6.out bootable */
	FPOWER64,		/* 9.out */
	FPOWER64B,		/* 9.out bootable */
	FWINPE,			/* windows PE executable */

	ANONE = 0,		/* dissembler types */
	AMIPS,
	AMIPSCO,		/* native mips */
	ASPARC,
	ASUNSPARC,		/* native sun */
	A68020,
	AI386,
	AI8086,			/* oh god */
	AI960,			/* retired */
	A29000,			/* retired */
	AARM,
	APOWER,
	AALPHA,
	ASPARC64,
	AAMD64,
	APOWER64,
				/* object file types */
	Obj68020 = 0,		/* .2 */
	ObjSparc,		/* .k */
	ObjMips,		/* .v */
	Obj386,			/* .8 */
	Obj960,			/* retired */
	Obj3210,		/* retired */
	ObjMips2,		/* .4 */
	Obj29000,		/* retired */
	ObjArm,			/* .5 */
	ObjPower,		/* .q */
	ObjMips2le,		/* .0 */
	ObjAlpha,		/* .7 */
	ObjSparc64,		/* .u */
	ObjAmd64,		/* .6 */
	ObjSpim,		/* .0 */
	ObjPower64,		/* .9 */
	Maxobjtype,

	CNONE  = 0,		/* symbol table classes */
	CAUTO,
	CPARAM,
	CSTAB,
	CTEXT,
	CDATA,
	CANY,			/* to look for any class */
};

typedef	struct	Map	Map;
typedef	struct	Symbol	Symbol;
typedef	struct	Reglist	Reglist;
typedef	struct	Mach	Mach;
typedef	struct	Machdata Machdata;
typedef	struct	Seg	Seg;

typedef int Maprw(Map *m, Seg *s, uvlong addr, void *v, uint n, int isread);

struct Seg {
	char	*name;		/* the segment name */
	int	fd;		/* file descriptor */
	int	inuse;		/* in use - not in use */
	int	cache;		/* should cache reads? */
	uvlong	b;		/* base */
	uvlong	e;		/* end */
	vlong	f;		/* offset within file */
	Maprw	*rw;		/* read/write fn for seg */
};

/*
 * 	Structure to map a segment to data
 */
struct Map {
	int	pid;
	int	tid;
	int	nsegs;	/* number of segments */
	Seg	seg[1];	/* actually n of these */
};

/*
 *	Internal structure describing a symbol table entry
 */
struct Symbol {
	void 	*handle;		/* used internally - owning func */
	struct {
		char	*name;
		vlong	value;		/* address or stack offset */
		char	type;		/* as in a.out.h */
		char	class;		/* as above */
		int	index;		/* in findlocal, globalsym, textsym */
	};
};

/*
 *	machine register description
 */
struct Reglist {
	char	*rname;			/* register name */
	short	roffs;			/* offset in u-block */
	char	rflags;			/* INTEGER/FLOAT, WRITABLE */
	char	rformat;		/* print format: 'x', 'X', 'f', '8', '3', 'Y', 'W' */
};

enum {					/* bits in rflags field */
	RINT	= (0<<0),
	RFLT	= (1<<0),
	RRDONLY	= (1<<1),
};

/*
 *	Machine-dependent data is stored in two structures:
 *		Mach  - miscellaneous general parameters
 *		Machdata - jump vector of service functions used by debuggers
 *
 *	Mach is defined in ?.c and set in executable.c
 *
 *	Machdata is defined in ?db.c
 *		and set in the debugger startup.
 */
struct Mach{
	char	*name;
	int	mtype;			/* machine type code */
	Reglist *reglist;		/* register set */
	int32	regsize;		/* sizeof registers in bytes */
	int32	fpregsize;		/* sizeof fp registers in bytes */
	char	*pc;			/* pc name */
	char	*sp;			/* sp name */
	char	*link;			/* link register name */
	char	*sbreg;			/* static base register name */
	uvlong	sb;			/* static base register value */
	int	pgsize;			/* page size */
	uvlong	kbase;			/* kernel base address */
	uvlong	ktmask;			/* ktzero = kbase & ~ktmask */
	uvlong	utop;			/* user stack top */
	int	pcquant;		/* quantization of pc */
	int	szaddr;			/* sizeof(void*) */
	int	szreg;			/* sizeof(register) */
	int	szfloat;		/* sizeof(float) */
	int	szdouble;		/* sizeof(double) */
};

extern	Mach	*mach;			/* Current machine */

typedef uvlong	(*Rgetter)(Map*, char*);
typedef	void	(*Tracer)(Map*, uvlong, uvlong, Symbol*);

struct	Machdata {		/* Machine-dependent debugger support */
	uchar	bpinst[4];			/* break point instr. */
	short	bpsize;				/* size of break point instr. */

	ushort	(*swab)(ushort);		/* ushort to local byte order */
	uint32	(*swal)(uint32);			/* uint32 to local byte order */
	uvlong	(*swav)(uvlong);		/* uvlong to local byte order */
	int	(*ctrace)(Map*, uvlong, uvlong, uvlong, Tracer); /* C traceback */
	uvlong	(*findframe)(Map*, uvlong, uvlong, uvlong, uvlong);/* frame finder */
	char*	(*excep)(Map*, Rgetter);	/* last exception */
	uint32	(*bpfix)(uvlong);		/* breakpoint fixup */
	int	(*sftos)(char*, int, void*);	/* single precision float */
	int	(*dftos)(char*, int, void*);	/* double precision float */
	int	(*foll)(Map*, uvlong, Rgetter, uvlong*);/* follow set */
	int	(*das)(Map*, uvlong, char, char*, int);	/* symbolic disassembly */
	int	(*hexinst)(Map*, uvlong, char*, int); 	/* hex disassembly */
	int	(*instsize)(Map*, uvlong);	/* instruction size */
};

/*
 *	Common a.out header describing all architectures
 */
typedef struct Fhdr
{
	char	*name;		/* identifier of executable */
	uchar	type;		/* file type - see codes above */
	uchar	hdrsz;		/* header size */
	uchar	_magic;		/* _MAGIC() magic */
	uchar	spare;
	int32	magic;		/* magic number */
	uvlong	txtaddr;	/* text address */
	vlong	txtoff;		/* start of text in file */
	uvlong	dataddr;	/* start of data segment */
	vlong	datoff;		/* offset to data seg in file */
	vlong	symoff;		/* offset of symbol table in file */
	uvlong	entry;		/* entry point */
	vlong	sppcoff;	/* offset of sp-pc table in file */
	vlong	lnpcoff;	/* offset of line number-pc table in file */
	int32	txtsz;		/* text size */
	int32	datsz;		/* size of data seg */
	int32	bsssz;		/* size of bss */
	int32	symsz;		/* size of symbol table */
	int32	sppcsz;		/* size of sp-pc table */
	int32	lnpcsz;		/* size of line number-pc table */
} Fhdr;

extern	int	asstype;	/* dissembler type - machdata.c */
extern	Machdata *machdata;	/* jump vector - machdata.c */

int		beieee80ftos(char*, int, void*);
int		beieeesftos(char*, int, void*);
int		beieeedftos(char*, int, void*);
ushort		beswab(ushort);
uint32		beswal(uint32);
uvlong		beswav(uvlong);
uvlong		ciscframe(Map*, uvlong, uvlong, uvlong, uvlong);
int		cisctrace(Map*, uvlong, uvlong, uvlong, Tracer);
int		crackhdr(int fd, Fhdr*);
uvlong		file2pc(char*, int32);
int		fileelem(Sym**, uchar *, char*, int);
int32		fileline(char*, int, uvlong);
int		filesym(int, char*, int);
int		findlocal(Symbol*, char*, Symbol*);
int		findseg(Map*, char*);
int		findsym(uvlong, int, Symbol *);
int		fnbound(uvlong, uvlong*);
int		fpformat(Map*, Reglist*, char*, int, int);
int		get1(Map*, uvlong, uchar*, int);
int		get2(Map*, uvlong, ushort*);
int		get4(Map*, uvlong, uint32*);
int		get8(Map*, uvlong, uvlong*);
int		geta(Map*, uvlong, uvlong*);
int		getauto(Symbol*, int, int, Symbol*);
Sym*		getsym(int);
int		globalsym(Symbol *, int);
char*		_hexify(char*, uint32, int);
int		ieeesftos(char*, int, uint32);
int		ieeedftos(char*, int, uint32, uint32);
int		isar(Biobuf*);
int		leieee80ftos(char*, int, void*);
int		leieeesftos(char*, int, void*);
int		leieeedftos(char*, int, void*);
ushort		leswab(ushort);
uint32		leswal(uint32);
uvlong		leswav(uvlong);
uvlong		line2addr(int32, uvlong, uvlong);
Map*		loadmap(Map*, int, Fhdr*);
int		localaddr(Map*, char*, char*, uvlong*, Rgetter);
int		localsym(Symbol*, int);
int		lookup(char*, char*, Symbol*);
void		machbytype(int);
int		machbyname(char*);
int		nextar(Biobuf*, int, char*);
Map*		newmap(Map*, int);
void		objtraverse(void(*)(Sym*, void*), void*);
int		objtype(Biobuf*, char**);
uvlong		pc2sp(uvlong);
int32		pc2line(uvlong);
int		put1(Map*, uvlong, uchar*, int);
int		put2(Map*, uvlong, ushort);
int		put4(Map*, uvlong, uint32);
int		put8(Map*, uvlong, uvlong);
int		puta(Map*, uvlong, uvlong);
int		readar(Biobuf*, int, vlong, int);
int		readobj(Biobuf*, int);
uvlong		riscframe(Map*, uvlong, uvlong, uvlong, uvlong);
int		risctrace(Map*, uvlong, uvlong, uvlong, Tracer);
int		setmap(Map*, int, uvlong, uvlong, vlong, char*, Maprw *rw);
Sym*		symbase(int32*);
int		syminit(int, Fhdr*);
int		symoff(char*, int, uvlong, int);
void		textseg(uvlong, Fhdr*);
int		textsym(Symbol*, int);
void		unusemap(Map*, int);

Map*		attachproc(int pid, Fhdr *fp);
int		ctlproc(int pid, char *msg);
void		detachproc(Map *m);
int		procnotes(int pid, char ***pnotes);
char*		proctextfile(int pid);
int		procthreadpids(int pid, int *tid, int ntid);
char*	procstatus(int);

Maprw	fdrw;
