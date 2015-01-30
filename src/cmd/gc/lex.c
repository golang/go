// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include	<u.h>
#include	<libc.h>
#include	"go.h"
#include	"y.tab.h"
#include	<ar.h>

#ifndef PLAN9
#include	<signal.h>
#endif

#undef	getc
#undef	ungetc
#define	getc	ccgetc
#define	ungetc	ccungetc

extern int yychar;
int yyprev;
int yylast;

static int	imported_unsafe;

static void	lexinit(void);
static void	lexinit1(void);
static void	lexfini(void);
static void	yytinit(void);
static int	getc(void);
static void	ungetc(int);
static int32	getr(void);
static int	escchar(int, int*, vlong*);
static void	addidir(char*);
static int	getlinepragma(void);
static char *goos, *goarch, *goroot;

#define	BOM	0xFEFF

// Debug arguments.
// These can be specified with the -d flag, as in "-d nil"
// to set the debug_checknil variable. In general the list passed
// to -d can be comma-separated.
static struct {
	char *name;
	int *val;
} debugtab[] = {
	{"nil", &debug_checknil},
};

// Our own isdigit, isspace, isalpha, isalnum that take care 
// of EOF and other out of range arguments.
static int
yy_isdigit(int c)
{
	return c >= 0 && c <= 0xFF && isdigit(c);
}

static int
yy_isspace(int c)
{
	return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

static int
yy_isalpha(int c)
{
	return c >= 0 && c <= 0xFF && isalpha(c);
}

static int
yy_isalnum(int c)
{
	return c >= 0 && c <= 0xFF && isalnum(c);
}

// Disallow use of isdigit etc.
#undef isdigit
#undef isspace
#undef isalpha
#undef isalnum
#define isdigit use_yy_isdigit_instead_of_isdigit
#define isspace use_yy_isspace_instead_of_isspace
#define isalpha use_yy_isalpha_instead_of_isalpha
#define isalnum use_yy_isalnum_instead_of_isalnum

#define	DBG	if(!debug['x']){}else print
/*c2go void DBG(char*, ...); */

enum
{
	EOF		= -1,
};

void
usage(void)
{
	print("usage: %cg [options] file.go...\n", arch.thechar);
	flagprint(1);
	exits("usage");
}

void
fault(int s)
{
	USED(s);

	// If we've already complained about things
	// in the program, don't bother complaining
	// about the seg fault too; let the user clean up
	// the code and try again.
	if(nsavederrors + nerrors > 0)
		errorexit();
	fatal("fault");
}

#ifdef	PLAN9
void
catcher(void *v, char *s)
{
	USED(v);

	if(strncmp(s, "sys: trap: fault read", 21) == 0) {
		if(nsavederrors + nerrors > 0)
			errorexit();
		fatal("fault");
	}
	noted(NDFLT);
}
#endif

void
doversion(void)
{
	char *p, *sep;

	p = expstring();
	if(strcmp(p, "X:none") == 0)
		p = "";
	sep = "";
	if(*p)
		sep = " ";
	print("%cg version %s%s%s\n", arch.thechar, getgoversion(), sep, p);
	exits(0);
}

int
gcmain(int argc, char *argv[])
{
	int i;
	NodeList *l;
	char *p;
	
#ifdef	SIGBUS	
	signal(SIGBUS, fault);
	signal(SIGSEGV, fault);
#endif

#ifdef	PLAN9
	notify(catcher);
	// Tell the FPU to handle all exceptions.
	setfcr(FPPDBL|FPRNR);
#endif
	// Allow GOARCH=arch.thestring or GOARCH=arch.thestringsuffix,
	// but not other values.	
	p = getgoarch();
	if(strncmp(p, arch.thestring, strlen(arch.thestring)) != 0)
		sysfatal("cannot use %cg with GOARCH=%s", arch.thechar, p);
	goarch = p;

	arch.linkarchinit();
	ctxt = linknew(arch.thelinkarch);
	ctxt->diag = yyerror;
	ctxt->bso = &bstdout;
	Binit(&bstdout, 1, OWRITE);

	localpkg = mkpkg(newstrlit(""));
	localpkg->prefix = "\"\"";
	
	// pseudo-package, for scoping
	builtinpkg = mkpkg(newstrlit("go.builtin"));

	// pseudo-package, accessed by import "unsafe"
	unsafepkg = mkpkg(newstrlit("unsafe"));
	unsafepkg->name = "unsafe";

	// real package, referred to by generated runtime calls
	runtimepkg = mkpkg(newstrlit("runtime"));
	runtimepkg->name = "runtime";

	// pseudo-packages used in symbol tables
	gostringpkg = mkpkg(newstrlit("go.string"));
	gostringpkg->name = "go.string";
	gostringpkg->prefix = "go.string";	// not go%2estring

	itabpkg = mkpkg(newstrlit("go.itab"));
	itabpkg->name = "go.itab";
	itabpkg->prefix = "go.itab";	// not go%2eitab

	weaktypepkg = mkpkg(newstrlit("go.weak.type"));
	weaktypepkg->name = "go.weak.type";
	weaktypepkg->prefix = "go.weak.type";  // not go%2eweak%2etype
	
	typelinkpkg = mkpkg(newstrlit("go.typelink"));
	typelinkpkg->name = "go.typelink";
	typelinkpkg->prefix = "go.typelink"; // not go%2etypelink

	trackpkg = mkpkg(newstrlit("go.track"));
	trackpkg->name = "go.track";
	trackpkg->prefix = "go.track";  // not go%2etrack

	typepkg = mkpkg(newstrlit("type"));
	typepkg->name = "type";

	goroot = getgoroot();
	goos = getgoos();

	nacl = strcmp(goos, "nacl") == 0;
	if(nacl)
		flag_largemodel = 1;

	fmtstrinit(&pragcgobuf);
	quotefmtinstall();

	outfile = nil;
	flagcount("+", "compiling runtime", &compiling_runtime);
	flagcount("%", "debug non-static initializers", &debug['%']);
	flagcount("A", "for bootstrapping, allow 'any' type", &debug['A']);
	flagcount("B", "disable bounds checking", &debug['B']);
	flagstr("D", "path: set relative path for local imports", &localimport);
	flagcount("E", "debug symbol export", &debug['E']);
	flagfn1("I", "dir: add dir to import search path", addidir);
	flagcount("K", "debug missing line numbers", &debug['K']);
	flagcount("L", "use full (long) path in error messages", &debug['L']);
	flagcount("M", "debug move generation", &debug['M']);
	flagcount("N", "disable optimizations", &debug['N']);
	flagcount("P", "debug peephole optimizer", &debug['P']);
	flagcount("R", "debug register optimizer", &debug['R']);
	flagcount("S", "print assembly listing", &debug['S']);
	flagfn0("V", "print compiler version", doversion);
	flagcount("W", "debug parse tree after type checking", &debug['W']);
	flagstr("asmhdr", "file: write assembly header to named file", &asmhdr);
	flagcount("complete", "compiling complete package (no C or assembly)", &pure_go);
	flagstr("d", "list: print debug information about items in list", &debugstr);
	flagcount("e", "no limit on number of errors reported", &debug['e']);
	flagcount("f", "debug stack frames", &debug['f']);
	flagcount("g", "debug code generation", &debug['g']);
	flagcount("h", "halt on error", &debug['h']);
	flagcount("i", "debug line number stack", &debug['i']);
	flagstr("installsuffix", "pkg directory suffix", &flag_installsuffix);
	flagcount("j", "debug runtime-initialized variables", &debug['j']);
	flagcount("l", "disable inlining", &debug['l']);
	flagcount("live", "debug liveness analysis", &debuglive);
	flagcount("m", "print optimization decisions", &debug['m']);
	flagcount("nolocalimports", "reject local (relative) imports", &nolocalimports);
	flagstr("o", "obj: set output file", &outfile);
	flagstr("p", "path: set expected package import path", &myimportpath);
	flagcount("pack", "write package file instead of object file", &writearchive);
	flagcount("r", "debug generated wrappers", &debug['r']);
	flagcount("race", "enable race detector", &flag_race);
	flagcount("s", "warn about composite literals that can be simplified", &debug['s']);
	flagstr("trimpath", "prefix: remove prefix from recorded source file paths", &ctxt->trimpath);
	flagcount("u", "reject unsafe code", &safemode);
	flagcount("v", "increase debug verbosity", &debug['v']);
	flagcount("w", "debug type checking", &debug['w']);
	use_writebarrier = 1;
	flagcount("wb", "enable write barrier", &use_writebarrier);
	flagcount("x", "debug lexer", &debug['x']);
	flagcount("y", "debug declarations in canned imports (with -d)", &debug['y']);
	if(arch.thechar == '6')
		flagcount("largemodel", "generate code that assumes a large memory model", &flag_largemodel);

	flagparse(&argc, &argv, usage);
	ctxt->debugasm = debug['S'];
	ctxt->debugvlog = debug['v'];

	if(argc < 1)
		usage();

	if(flag_race) {
		racepkg = mkpkg(newstrlit("runtime/race"));
		racepkg->name = "race";
	}
	
	// parse -d argument
	if(debugstr) {
		char *f[100];
		int i, j, nf;
		
		nf = getfields(debugstr, f, nelem(f), 1, ",");
		for(i=0; i<nf; i++) {
			for(j=0; j<nelem(debugtab); j++) {
				if(strcmp(debugtab[j].name, f[i]) == 0) {
					if(debugtab[j].val != nil)
						*debugtab[j].val = 1;
					break;
				}
			}
			if(j >= nelem(debugtab))
				sysfatal("unknown debug information -d '%s'\n", f[i]);
		}
	}

	// enable inlining.  for now:
	//	default: inlining on.  (debug['l'] == 1)
	//	-l: inlining off  (debug['l'] == 0)
	//	-ll, -lll: inlining on again, with extra debugging (debug['l'] > 1)
	if(debug['l'] <= 1)
		debug['l'] = 1 - debug['l'];

	if(arch.thechar == '8') {
		p = getgo386();
		if(strcmp(p, "387") == 0)
			use_sse = 0;
		else if(strcmp(p, "sse2") == 0)
			use_sse = 1;
		else
			sysfatal("unsupported setting GO386=%s", p);
	}

	fmtinstallgo();
	arch.betypeinit();
	if(widthptr == 0)
		fatal("betypeinit failed");

	lexinit();
	typeinit();
	lexinit1();
	yytinit();

	blockgen = 1;
	dclcontext = PEXTERN;
	nerrors = 0;
	lexlineno = 1;

	for(i=0; i<argc; i++) {
		infile = argv[i];
		linehist(infile, 0, 0);

		curio.infile = infile;
		curio.bin = Bopen(infile, OREAD);
		if(curio.bin == nil) {
			print("open %s: %r\n", infile);
			errorexit();
		}
		curio.peekc = 0;
		curio.peekc1 = 0;
		curio.nlsemi = 0;
		curio.eofnl = 0;
		curio.last = 0;

		// Skip initial BOM if present.
		if(Bgetrune(curio.bin) != BOM)
			Bungetrune(curio.bin);

		block = 1;
		iota = -1000000;
		
		imported_unsafe = 0;

		yyparse();
		if(nsyntaxerrors != 0)
			errorexit();

		linehist(nil, 0, 0);
		if(curio.bin != nil)
			Bterm(curio.bin);
	}
	testdclstack();
	mkpackage(localpkg->name);	// final import not used checks
	lexfini();

	typecheckok = 1;
	if(debug['f'])
		frame(1);

	// Process top-level declarations in phases.

	// Phase 1: const, type, and names and types of funcs.
	//   This will gather all the information about types
	//   and methods but doesn't depend on any of it.
	defercheckwidth();
	for(l=xtop; l; l=l->next)
		if(l->n->op != ODCL && l->n->op != OAS)
			typecheck(&l->n, Etop);

	// Phase 2: Variable assignments.
	//   To check interface assignments, depends on phase 1.
	for(l=xtop; l; l=l->next)
		if(l->n->op == ODCL || l->n->op == OAS)
			typecheck(&l->n, Etop);
	resumecheckwidth();

	// Phase 3: Type check function bodies.
	for(l=xtop; l; l=l->next) {
		if(l->n->op == ODCLFUNC || l->n->op == OCLOSURE) {
			curfn = l->n;
			decldepth = 1;
			saveerrors();
			typechecklist(l->n->nbody, Etop);
			checkreturn(l->n);
			if(nerrors != 0)
				l->n->nbody = nil;  // type errors; do not compile
		}
	}

	// Phase 4: Decide how to capture variables
	// and transform closure bodies accordingly.
	for(l=xtop; l; l=l->next) {
		if(l->n->op == ODCLFUNC && l->n->closure) {
			curfn = l->n;
			capturevars(l->n);
		}
	}

	curfn = nil;
	
	if(nsavederrors+nerrors)
		errorexit();

	// Phase 5: Inlining
	if(debug['l'] > 1) {
		// Typecheck imported function bodies if debug['l'] > 1,
		// otherwise lazily when used or re-exported.
		for(l=importlist; l; l=l->next)
			if (l->n->inl) {
				saveerrors();
				typecheckinl(l->n);
			}
		
		if(nsavederrors+nerrors)
			errorexit();
	}

	if(debug['l']) {
		// Find functions that can be inlined and clone them before walk expands them.
		for(l=xtop; l; l=l->next)
			if(l->n->op == ODCLFUNC)
				caninl(l->n);
		
		// Expand inlineable calls in all functions
		for(l=xtop; l; l=l->next)
			if(l->n->op == ODCLFUNC)
				inlcalls(l->n);
	}

	// Phase 6: Escape analysis.
	// Required for moving heap allocations onto stack,
	// which in turn is required by the closure implementation,
	// which stores the addresses of stack variables into the closure.
	// If the closure does not escape, it needs to be on the stack
	// or else the stack copier will not update it.
	escapes(xtop);
	
	// Escape analysis moved escaped values off stack.
	// Move large values off stack too.
	movelarge(xtop);

	// Phase 7: Compile top level functions.
	for(l=xtop; l; l=l->next)
		if(l->n->op == ODCLFUNC)
			funccompile(l->n, 0);

	if(nsavederrors+nerrors == 0)
		fninit(xtop);

	// Phase 8: Check external declarations.
	for(l=externdcl; l; l=l->next)
		if(l->n->op == ONAME)
			typecheck(&l->n, Erv);

	if(nerrors+nsavederrors)
		errorexit();

	dumpobj();
	
	if(asmhdr)
		dumpasmhdr();

	if(nerrors+nsavederrors)
		errorexit();

	flusherrors();
	exits(0);
	return 0;
}

void
saveerrors(void)
{
	nsavederrors += nerrors;
	nerrors = 0;
}

static int
arsize(Biobuf *b, char *name)
{
	struct ar_hdr a;

	if(Bread(b, a.name, sizeof(a.name)) != sizeof(a.name) ||
	   Bread(b, a.date, sizeof(a.date)) != sizeof(a.date) ||
	   Bread(b, a.uid, sizeof(a.uid)) != sizeof(a.uid) ||
	   Bread(b, a.gid, sizeof(a.gid)) != sizeof(a.gid) ||
	   Bread(b, a.mode, sizeof(a.mode)) != sizeof(a.mode) ||
	   Bread(b, a.size, sizeof(a.size)) != sizeof(a.size) ||
	   Bread(b, a.fmag, sizeof(a.fmag)) != sizeof(a.fmag))
		return -1;

	if(strncmp(a.name, name, strlen(name)) != 0)
		return -1;

	return atoi(a.size);
}

static int
skiptopkgdef(Biobuf *b)
{
	char *p;
	int sz;

	/* archive header */
	if((p = Brdline(b, '\n')) == nil)
		return 0;
	if(Blinelen(b) != 8)
		return 0;
	if(memcmp(p, "!<arch>\n", 8) != 0)
		return 0;
	/* symbol table may be first; skip it */
	sz = arsize(b, "__.GOSYMDEF");
	if(sz >= 0)
		Bseek(b, sz, 1);
	else
		Bseek(b, 8, 0);
	/* package export block is next */
	sz = arsize(b, "__.PKGDEF");
	if(sz <= 0)
		return 0;
	return 1;
}

static void
addidir(char* dir)
{
	Idir** pp;

	if(dir == nil)
		return;

	for(pp = &idirs; *pp != nil; pp = &(*pp)->link)
		;
	*pp = mal(sizeof(Idir));
	(*pp)->link = nil;
	(*pp)->dir = dir;
}

// is this path a local name?  begins with ./ or ../ or /
static int
islocalname(Strlit *name)
{
	if(name->len >= 1 && name->s[0] == '/')
		return 1;
	if(ctxt->windows && name->len >= 3 &&
	   yy_isalpha(name->s[0]) && name->s[1] == ':' && name->s[2] == '/')
	   	return 1;
	if(name->len >= 2 && strncmp(name->s, "./", 2) == 0)
		return 1;
	if(name->len == 1 && strncmp(name->s, ".", 1) == 0)
		return 1;
	if(name->len >= 3 && strncmp(name->s, "../", 3) == 0)
		return 1;
	if(name->len == 2 && strncmp(name->s, "..", 2) == 0)
		return 1;
	return 0;
}

static int
findpkg(Strlit *name)
{
	Idir *p;
	char *q, *suffix, *suffixsep;

	if(islocalname(name)) {
		if(safemode || nolocalimports)
			return 0;
		// try .a before .6.  important for building libraries:
		// if there is an array.6 in the array.a library,
		// want to find all of array.a, not just array.6.
		snprint(namebuf, sizeof(namebuf), "%Z.a", name);
		if(access(namebuf, 0) >= 0)
			return 1;
		snprint(namebuf, sizeof(namebuf), "%Z.%c", name, arch.thechar);
		if(access(namebuf, 0) >= 0)
			return 1;
		return 0;
	}

	// local imports should be canonicalized already.
	// don't want to see "encoding/../encoding/base64"
	// as different from "encoding/base64".
	q = mal(name->len+1);
	memmove(q, name->s, name->len);
	q[name->len] = '\0';
	cleanname(q);
	if(strlen(q) != name->len || memcmp(q, name->s, name->len) != 0) {
		yyerror("non-canonical import path %Z (should be %s)", name, q);
		return 0;
	}

	for(p = idirs; p != nil; p = p->link) {
		snprint(namebuf, sizeof(namebuf), "%s/%Z.a", p->dir, name);
		if(access(namebuf, 0) >= 0)
			return 1;
		snprint(namebuf, sizeof(namebuf), "%s/%Z.%c", p->dir, name, arch.thechar);
		if(access(namebuf, 0) >= 0)
			return 1;
	}
	if(goroot != nil) {
		suffix = "";
		suffixsep = "";
		if(flag_installsuffix != nil) {
			suffixsep = "_";
			suffix = flag_installsuffix;
		} else if(flag_race) {
			suffixsep = "_";
			suffix = "race";
		}
		snprint(namebuf, sizeof(namebuf), "%s/pkg/%s_%s%s%s/%Z.a", goroot, goos, goarch, suffixsep, suffix, name);
		if(access(namebuf, 0) >= 0)
			return 1;
		snprint(namebuf, sizeof(namebuf), "%s/pkg/%s_%s%s%s/%Z.%c", goroot, goos, goarch, suffixsep, suffix, name, arch.thechar);
		if(access(namebuf, 0) >= 0)
			return 1;
	}
	return 0;
}

static void
fakeimport(void)
{
	importpkg = mkpkg(newstrlit("fake"));
	cannedimports("fake.6", "$$\n");
}

void
importfile(Val *f, int line)
{
	Biobuf *imp;
	char *file, *p, *q, *tag;
	int32 c;
	int len;
	Strlit *path;
	char *cleanbuf, *prefix;

	USED(line);

	if(f->ctype != CTSTR) {
		yyerror("import statement not a string");
		fakeimport();
		return;
	}

	if(f->u.sval->len == 0) {
		yyerror("import path is empty");
		fakeimport();
		return;
	}

	if(isbadimport(f->u.sval)) {
		fakeimport();
		return;
	}

	// The package name main is no longer reserved,
	// but we reserve the import path "main" to identify
	// the main package, just as we reserve the import 
	// path "math" to identify the standard math package.
	if(strcmp(f->u.sval->s, "main") == 0) {
		yyerror("cannot import \"main\"");
		errorexit();
	}

	if(myimportpath != nil && strcmp(f->u.sval->s, myimportpath) == 0) {
		yyerror("import \"%Z\" while compiling that package (import cycle)", f->u.sval);
		errorexit();
	}

	if(strcmp(f->u.sval->s, "unsafe") == 0) {
		if(safemode) {
			yyerror("cannot import package unsafe");
			errorexit();
		}
		importpkg = mkpkg(f->u.sval);
		cannedimports("unsafe.6", unsafeimport);
		imported_unsafe = 1;
		return;
	}
	
	path = f->u.sval;
	if(islocalname(path)) {
		if(path->s[0] == '/') {
			yyerror("import path cannot be absolute path");
			fakeimport();
			return;
		}
		prefix = ctxt->pathname;
		if(localimport != nil)
			prefix = localimport;
		cleanbuf = mal(strlen(prefix) + strlen(path->s) + 2);
		strcpy(cleanbuf, prefix);
		strcat(cleanbuf, "/");
		strcat(cleanbuf, path->s);
		cleanname(cleanbuf);
		path = newstrlit(cleanbuf);
		
		if(isbadimport(path)) {
			fakeimport();
			return;
		}
	}

	if(!findpkg(path)) {
		yyerror("can't find import: \"%Z\"", f->u.sval);
		errorexit();
	}
	importpkg = mkpkg(path);

	// If we already saw that package, feed a dummy statement
	// to the lexer to avoid parsing export data twice.
	if(importpkg->imported) {
		file = strdup(namebuf);
		tag = "";
		if(importpkg->safe) {
			tag = "safe";
		}
		p = smprint("package %s %s\n$$\n", importpkg->name, tag);
		cannedimports(file, p);
		return;
	}
	importpkg->imported = 1;

	imp = Bopen(namebuf, OREAD);
	if(imp == nil) {
		yyerror("can't open import: \"%Z\": %r", f->u.sval);
		errorexit();
	}
	file = strdup(namebuf);

	len = strlen(namebuf);
	if(len > 2 && namebuf[len-2] == '.' && namebuf[len-1] == 'a') {
		if(!skiptopkgdef(imp)) {
			yyerror("import %s: not a package file", file);
			errorexit();
		}
	}
	
	// check object header
	p = Brdstr(imp, '\n', 1);
	if(strcmp(p, "empty archive") != 0) {
		if(strncmp(p, "go object ", 10) != 0) {
			yyerror("import %s: not a go object file", file);
			errorexit();
		}
		q = smprint("%s %s %s %s", getgoos(), getgoarch(), getgoversion(), expstring());
		if(strcmp(p+10, q) != 0) {
			yyerror("import %s: object is [%s] expected [%s]", file, p+10, q);
			errorexit();
		}
		free(q);
	}

	// assume files move (get installed)
	// so don't record the full path.
	linehist(file + len - path->len - 2, -1, 1);	// acts as #pragma lib

	/*
	 * position the input right
	 * after $$ and return
	 */
	pushedio = curio;
	curio.bin = imp;
	curio.peekc = 0;
	curio.peekc1 = 0;
	curio.infile = file;
	curio.nlsemi = 0;
	typecheckok = 1;

	for(;;) {
		c = getc();
		if(c == EOF)
			break;
		if(c != '$')
			continue;
		c = getc();
		if(c == EOF)
			break;
		if(c != '$')
			continue;
		return;
	}
	yyerror("no import in \"%Z\"", f->u.sval);
	unimportfile();
}

void
unimportfile(void)
{
	if(curio.bin != nil) {
		Bterm(curio.bin);
		curio.bin = nil;
	} else
		lexlineno--;	// re correct sys.6 line number

	curio = pushedio;
	pushedio.bin = nil;
	incannedimport = 0;
	typecheckok = 0;
}

void
cannedimports(char *file, char *cp)
{
	lexlineno++;		// if sys.6 is included on line 1,

	pushedio = curio;
	curio.bin = nil;
	curio.peekc = 0;
	curio.peekc1 = 0;
	curio.infile = file;
	curio.cp = cp;
	curio.nlsemi = 0;
	curio.importsafe = 0;

	typecheckok = 1;
	incannedimport = 1;
}

static int
isfrog(int c)
{
	// complain about possibly invisible control characters
	if(c < ' ') {
		return !yy_isspace(c);	// exclude good white space
	}
	if(0x7f <= c && c <= 0xa0)	// DEL, unicode block including unbreakable space.
		return 1;
	return 0;
}

typedef struct Loophack Loophack;
struct Loophack {
	int v;
	Loophack *next;
};

static int32
_yylex(void)
{
	int c, c1, clen, escflag, ncp;
	vlong v;
	char *cp, *ep;
	Rune rune;
	Sym *s;
	static Loophack *lstk;
	Loophack *h;

	prevlineno = lineno;

l0:
	c = getc();
	if(yy_isspace(c)) {
		if(c == '\n' && curio.nlsemi) {
			ungetc(c);
			DBG("lex: implicit semi\n");
			return ';';
		}
		goto l0;
	}

	lineno = lexlineno;	/* start of token */

	if(c >= Runeself) {
		/* all multibyte runes are alpha */
		cp = lexbuf;
		ep = lexbuf+sizeof lexbuf;
		goto talph;
	}

	if(yy_isalpha(c)) {
		cp = lexbuf;
		ep = lexbuf+sizeof lexbuf;
		goto talph;
	}

	if(yy_isdigit(c))
		goto tnum;

	switch(c) {
	case EOF:
		lineno = prevlineno;
		ungetc(EOF);
		return -1;

	case '_':
		cp = lexbuf;
		ep = lexbuf+sizeof lexbuf;
		goto talph;

	case '.':
		c1 = getc();
		if(yy_isdigit(c1)) {
			cp = lexbuf;
			ep = lexbuf+sizeof lexbuf;
			*cp++ = c;
			c = c1;
			goto casedot;
		}
		if(c1 == '.') {
			c1 = getc();
			if(c1 == '.') {
				c = LDDD;
				goto lx;
			}
			ungetc(c1);
			c1 = '.';
		}
		break;

	case '"':
		/* "..." */
		strcpy(lexbuf, "\"<string>\"");
		cp = mal(8);
		clen = sizeof(int32);
		ncp = 8;

		for(;;) {
			if(clen+UTFmax > ncp) {
				cp = remal(cp, ncp, ncp);
				ncp += ncp;
			}
			if(escchar('"', &escflag, &v))
				break;
			if(v < Runeself || escflag) {
				cp[clen++] = v;
			} else {
				rune = v;
				c = runelen(rune);
				runetochar(cp+clen, &rune);
				clen += c;
			}
		}
		goto strlit;
	
	case '`':
		/* `...` */
		strcpy(lexbuf, "`<string>`");
		cp = mal(8);
		clen = sizeof(int32);
		ncp = 8;

		for(;;) {
			if(clen+UTFmax > ncp) {
				cp = remal(cp, ncp, ncp);
				ncp += ncp;
			}
			c = getr();
			if(c == '\r')
				continue;
			if(c == EOF) {
				yyerror("eof in string");
				break;
			}
			if(c == '`')
				break;
			rune = c;
			clen += runetochar(cp+clen, &rune);
		}

	strlit:
		*(int32*)cp = clen-sizeof(int32);	// length
		do {
			cp[clen++] = 0;
		} while(clen & MAXALIGN);
		yylval.val.u.sval = (Strlit*)cp;
		yylval.val.ctype = CTSTR;
		DBG("lex: string literal\n");
		strcpy(litbuf, "string literal");
		return LLITERAL;

	case '\'':
		/* '.' */
		if(escchar('\'', &escflag, &v)) {
			yyerror("empty character literal or unescaped ' in character literal");
			v = '\'';
		}
		if(!escchar('\'', &escflag, &v)) {
			yyerror("missing '");
			ungetc(v);
		}
		yylval.val.u.xval = mal(sizeof(*yylval.val.u.xval));
		mpmovecfix(yylval.val.u.xval, v);
		yylval.val.ctype = CTRUNE;
		DBG("lex: codepoint literal\n");
		strcpy(litbuf, "string literal");
		return LLITERAL;

	case '/':
		c1 = getc();
		if(c1 == '*') {
			int nl;
			
			nl = 0;
			for(;;) {
				c = getr();
				if(c == '\n')
					nl = 1;
				while(c == '*') {
					c = getr();
					if(c == '/') {
						if(nl)
							ungetc('\n');
						goto l0;
					}
					if(c == '\n')
						nl = 1;
				}
				if(c == EOF) {
					yyerror("eof in comment");
					errorexit();
				}
			}
		}
		if(c1 == '/') {
			c = getlinepragma();
			for(;;) {
				if(c == '\n' || c == EOF) {
					ungetc(c);
					goto l0;
				}
				c = getr();
			}
		}
		if(c1 == '=') {
			c = ODIV;
			goto asop;
		}
		break;

	case ':':
		c1 = getc();
		if(c1 == '=') {
			c = LCOLAS;
			yylval.i = lexlineno;
			goto lx;
		}
		break;

	case '*':
		c1 = getc();
		if(c1 == '=') {
			c = OMUL;
			goto asop;
		}
		break;

	case '%':
		c1 = getc();
		if(c1 == '=') {
			c = OMOD;
			goto asop;
		}
		break;

	case '+':
		c1 = getc();
		if(c1 == '+') {
			c = LINC;
			goto lx;
		}
		if(c1 == '=') {
			c = OADD;
			goto asop;
		}
		break;

	case '-':
		c1 = getc();
		if(c1 == '-') {
			c = LDEC;
			goto lx;
		}
		if(c1 == '=') {
			c = OSUB;
			goto asop;
		}
		break;

	case '>':
		c1 = getc();
		if(c1 == '>') {
			c = LRSH;
			c1 = getc();
			if(c1 == '=') {
				c = ORSH;
				goto asop;
			}
			break;
		}
		if(c1 == '=') {
			c = LGE;
			goto lx;
		}
		c = LGT;
		break;

	case '<':
		c1 = getc();
		if(c1 == '<') {
			c = LLSH;
			c1 = getc();
			if(c1 == '=') {
				c = OLSH;
				goto asop;
			}
			break;
		}
		if(c1 == '=') {
			c = LLE;
			goto lx;
		}
		if(c1 == '-') {
			c = LCOMM;
			goto lx;
		}
		c = LLT;
		break;

	case '=':
		c1 = getc();
		if(c1 == '=') {
			c = LEQ;
			goto lx;
		}
		break;

	case '!':
		c1 = getc();
		if(c1 == '=') {
			c = LNE;
			goto lx;
		}
		break;

	case '&':
		c1 = getc();
		if(c1 == '&') {
			c = LANDAND;
			goto lx;
		}
		if(c1 == '^') {
			c = LANDNOT;
			c1 = getc();
			if(c1 == '=') {
				c = OANDNOT;
				goto asop;
			}
			break;
		}
		if(c1 == '=') {
			c = OAND;
			goto asop;
		}
		break;

	case '|':
		c1 = getc();
		if(c1 == '|') {
			c = LOROR;
			goto lx;
		}
		if(c1 == '=') {
			c = OOR;
			goto asop;
		}
		break;

	case '^':
		c1 = getc();
		if(c1 == '=') {
			c = OXOR;
			goto asop;
		}
		break;

	/*
	 * clumsy dance:
	 * to implement rule that disallows
	 *	if T{1}[0] { ... }
	 * but allows
	 * 	if (T{1}[0]) { ... }
	 * the block bodies for if/for/switch/select
	 * begin with an LBODY token, not '{'.
	 *
	 * when we see the keyword, the next
	 * non-parenthesized '{' becomes an LBODY.
	 * loophack is normally 0.
	 * a keyword makes it go up to 1.
	 * parens push loophack onto a stack and go back to 0.
	 * a '{' with loophack == 1 becomes LBODY and disables loophack.
	 *
	 * i said it was clumsy.
	 */
	case '(':
	case '[':
		if(loophack || lstk != nil) {
			h = malloc(sizeof *h);
			if(h == nil) {
				flusherrors();
				yyerror("out of memory");
				errorexit();
			}
			h->v = loophack;
			h->next = lstk;
			lstk = h;
			loophack = 0;
		}
		goto lx;
	case ')':
	case ']':
		if(lstk != nil) {
			h = lstk;
			loophack = h->v;
			lstk = h->next;
			free(h);
		}
		goto lx;
	case '{':
		if(loophack == 1) {
			DBG("%L lex: LBODY\n", lexlineno);
			loophack = 0;
			return LBODY;
		}
		goto lx;

	default:
		goto lx;
	}
	ungetc(c1);

lx:
	if(c > 0xff)
		DBG("%L lex: TOKEN %s\n", lexlineno, lexname(c));
	else
		DBG("%L lex: TOKEN '%c'\n", lexlineno, c);
	if(isfrog(c)) {
		yyerror("illegal character 0x%ux", c);
		goto l0;
	}
	if(importpkg == nil && (c == '#' || c == '$' || c == '?' || c == '@' || c == '\\')) {
		yyerror("%s: unexpected %c", "syntax error", c);
		goto l0;
	}
	return c;

asop:
	yylval.i = c;	// rathole to hold which asop
	DBG("lex: TOKEN ASOP %c\n", c);
	return LASOP;

talph:
	/*
	 * cp is set to lexbuf and some
	 * prefix has been stored
	 */
	for(;;) {
		if(cp+10 >= ep) {
			yyerror("identifier too long");
			errorexit();
		}
		if(c >= Runeself) {
			ungetc(c);
			rune = getr();
			// 0xb7 Â· is used for internal names
			if(!isalpharune(rune) && !isdigitrune(rune) && (importpkg == nil || rune != 0xb7))
				yyerror("invalid identifier character U+%04x", rune);
			cp += runetochar(cp, &rune);
		} else if(!yy_isalnum(c) && c != '_')
			break;
		else
			*cp++ = c;
		c = getc();
	}
	*cp = 0;
	ungetc(c);

	s = lookup(lexbuf);
	switch(s->lexical) {
	case LIGNORE:
		goto l0;

	case LFOR:
	case LIF:
	case LSWITCH:
	case LSELECT:
		loophack = 1;	// see comment about loophack above
		break;
	}

	DBG("lex: %S %s\n", s, lexname(s->lexical));
	yylval.sym = s;
	return s->lexical;

tnum:
	cp = lexbuf;
	ep = lexbuf+sizeof lexbuf;
	if(c != '0') {
		for(;;) {
			if(cp+10 >= ep) {
				yyerror("identifier too long");
				errorexit();
			}
			*cp++ = c;
			c = getc();
			if(yy_isdigit(c))
				continue;
			goto dc;
		}
	}
	*cp++ = c;
	c = getc();
	if(c == 'x' || c == 'X') {
		for(;;) {
			if(cp+10 >= ep) {
				yyerror("identifier too long");
				errorexit();
			}
			*cp++ = c;
			c = getc();
			if(yy_isdigit(c))
				continue;
			if(c >= 'a' && c <= 'f')
				continue;
			if(c >= 'A' && c <= 'F')
				continue;
			if(cp == lexbuf+2)
				yyerror("malformed hex constant");
			if(c == 'p')
				goto caseep;
			goto ncu;
		}
	}

	if(c == 'p')	// 0p begins floating point zero
		goto caseep;

	c1 = 0;
	for(;;) {
		if(cp+10 >= ep) {
			yyerror("identifier too long");
			errorexit();
		}
		if(!yy_isdigit(c))
			break;
		if(c < '0' || c > '7')
			c1 = 1;		// not octal
		*cp++ = c;
		c = getc();
	}
	if(c == '.')
		goto casedot;
	if(c == 'e' || c == 'E')
		goto caseep;
	if(c == 'i')
		goto casei;
	if(c1)
		yyerror("malformed octal constant");
	goto ncu;

dc:
	if(c == '.')
		goto casedot;
	if(c == 'e' || c == 'E' || c == 'p' || c == 'P')
		goto caseep;
	if(c == 'i')
		goto casei;

ncu:
	*cp = 0;
	ungetc(c);

	yylval.val.u.xval = mal(sizeof(*yylval.val.u.xval));
	mpatofix(yylval.val.u.xval, lexbuf);
	if(yylval.val.u.xval->ovf) {
		yyerror("overflow in constant");
		mpmovecfix(yylval.val.u.xval, 0);
	}
	yylval.val.ctype = CTINT;
	DBG("lex: integer literal\n");
	strcpy(litbuf, "literal ");
	strcat(litbuf, lexbuf);
	return LLITERAL;

casedot:
	for(;;) {
		if(cp+10 >= ep) {
			yyerror("identifier too long");
			errorexit();
		}
		*cp++ = c;
		c = getc();
		if(!yy_isdigit(c))
			break;
	}
	if(c == 'i')
		goto casei;
	if(c != 'e' && c != 'E')
		goto caseout;

caseep:
	*cp++ = c;
	c = getc();
	if(c == '+' || c == '-') {
		*cp++ = c;
		c = getc();
	}
	if(!yy_isdigit(c))
		yyerror("malformed fp constant exponent");
	while(yy_isdigit(c)) {
		if(cp+10 >= ep) {
			yyerror("identifier too long");
			errorexit();
		}
		*cp++ = c;
		c = getc();
	}
	if(c == 'i')
		goto casei;
	goto caseout;

casei:
	// imaginary constant
	*cp = 0;
	yylval.val.u.cval = mal(sizeof(*yylval.val.u.cval));
	mpmovecflt(&yylval.val.u.cval->real, 0.0);
	mpatoflt(&yylval.val.u.cval->imag, lexbuf);
	if(yylval.val.u.cval->imag.val.ovf) {
		yyerror("overflow in imaginary constant");
		mpmovecflt(&yylval.val.u.cval->real, 0.0);
	}
	yylval.val.ctype = CTCPLX;
	DBG("lex: imaginary literal\n");
	strcpy(litbuf, "literal ");
	strcat(litbuf, lexbuf);
	return LLITERAL;

caseout:
	*cp = 0;
	ungetc(c);

	yylval.val.u.fval = mal(sizeof(*yylval.val.u.fval));
	mpatoflt(yylval.val.u.fval, lexbuf);
	if(yylval.val.u.fval->val.ovf) {
		yyerror("overflow in float constant");
		mpmovecflt(yylval.val.u.fval, 0.0);
	}
	yylval.val.ctype = CTFLT;
	DBG("lex: floating literal\n");
	strcpy(litbuf, "literal ");
	strcat(litbuf, lexbuf);
	return LLITERAL;
}

static void pragcgo(char*);

static int
more(char **pp)
{
	char *p;
	
	p = *pp;
	while(yy_isspace(*p))
		p++;
	*pp = p;
	return *p != '\0';
}

/*
 * read and interpret syntax that looks like
 * //line parse.y:15
 * as a discontinuity in sequential line numbers.
 * the next line of input comes from parse.y:15
 */
static int
getlinepragma(void)
{
	int i, c, n;
	char *cp, *ep, *linep;
	Hist *h;

	c = getr();
	if(c == 'g')
		goto go;
	if(c != 'l')	
		goto out;
	for(i=1; i<5; i++) {
		c = getr();
		if(c != "line "[i])
			goto out;
	}

	cp = lexbuf;
	ep = lexbuf+sizeof(lexbuf)-5;
	linep = nil;
	for(;;) {
		c = getr();
		if(c == EOF)
			goto out;
		if(c == '\n')
			break;
		if(c == ' ')
			continue;
		if(c == ':')
			linep = cp;
		if(cp < ep)
			*cp++ = c;
	}
	*cp = 0;

	if(linep == nil || linep >= ep)
		goto out;
	*linep++ = '\0';
	n = 0;
	for(cp=linep; *cp; cp++) {
		if(*cp < '0' || *cp > '9')
			goto out;
		n = n*10 + *cp - '0';
		if(n > 1e8) {
			yyerror("line number out of range");
			errorexit();
		}
	}
	if(n <= 0)
		goto out;

	// try to avoid allocating file name over and over
	for(h=ctxt->hist; h!=nil; h=h->link) {
		if(h->name != nil && strcmp(h->name, lexbuf) == 0) {
			linehist(h->name, n, 0);
			goto out;
		}
	}
	linehist(strdup(lexbuf), n, 0);
	goto out;

go:
	cp = lexbuf;
	ep = lexbuf+sizeof(lexbuf)-5;
	*cp++ = 'g'; // already read
	for(;;) {
		c = getr();
		if(c == EOF || c >= Runeself)
			goto out;
		if(c == '\n')
			break;
		if(cp < ep)
			*cp++ = c;
	}
	*cp = 0;
	
	if(strncmp(lexbuf, "go:cgo_", 7) == 0)
		pragcgo(lexbuf);
	
	ep = strchr(lexbuf, ' ');
	if(ep != nil)
		*ep = 0;
	
	if(strcmp(lexbuf, "go:linkname") == 0) {
		if(!imported_unsafe)
			yyerror("//go:linkname only allowed in Go files that import \"unsafe\"");
		if(ep == nil) {
			yyerror("usage: //go:linkname localname linkname");
			goto out;
		}
		cp = ep+1;
		while(yy_isspace(*cp))
			cp++;
		ep = strchr(cp, ' ');
		if(ep == nil) {
			yyerror("usage: //go:linkname localname linkname");
			goto out;
		}
		*ep++ = 0;
		while(yy_isspace(*ep))
			ep++;
		if(*ep == 0) {
			yyerror("usage: //go:linkname localname linkname");
			goto out;
		}
		lookup(cp)->linkname = strdup(ep);
		goto out;
	}	

	if(strcmp(lexbuf, "go:nointerface") == 0 && fieldtrack_enabled) {
		nointerface = 1;
		goto out;
	}
	if(strcmp(lexbuf, "go:noescape") == 0) {
		noescape = 1;
		goto out;
	}
	if(strcmp(lexbuf, "go:nosplit") == 0) {
		nosplit = 1;
		goto out;
	}
	if(strcmp(lexbuf, "go:nowritebarrier") == 0) {
		if(!compiling_runtime)
			yyerror("//go:nowritebarrier only allowed in runtime");
		nowritebarrier = 1;
		goto out;
	}
	
out:
	return c;
}

static char*
getimpsym(char **pp)
{
	char *p, *start;
	
	more(pp); // skip spaces

	p = *pp;
	if(*p == '\0' || *p == '"')
		return nil;
	
	start = p;
	while(*p != '\0' && !yy_isspace(*p) && *p != '"')
		p++;
	if(*p != '\0')
		*p++ = '\0';
	
	*pp = p;
	return start;
}

static char*
getquoted(char **pp)
{
	char *p, *start;
	
	more(pp); // skip spaces
	
	p = *pp;
	if(*p != '"')
		return nil;
	p++;
	
	start = p;
	while(*p != '"') {
		if(*p == '\0')
			return nil;
		p++;
	}
	*p++ = '\0';
	*pp = p;
	return start;
}

// Copied nearly verbatim from the C compiler's #pragma parser.
// TODO: Rewrite more cleanly once the compiler is written in Go.
static void
pragcgo(char *text)
{
	char *local, *remote, *p, *q, *verb;
	
	for(q=text; *q != '\0' && *q != ' '; q++)
		;
	if(*q == ' ')
		*q++ = '\0';
	
	verb = text+3; // skip "go:"

	if(strcmp(verb, "cgo_dynamic_linker") == 0 || strcmp(verb, "dynlinker") == 0) {
		p = getquoted(&q);
		if(p == nil)
			goto err1;
		fmtprint(&pragcgobuf, "cgo_dynamic_linker %q\n", p);
		goto out;
	
	err1:
		yyerror("usage: //go:cgo_dynamic_linker \"path\"");
		goto out;
	}	

	if(strcmp(verb, "dynexport") == 0)
		verb = "cgo_export_dynamic";
	if(strcmp(verb, "cgo_export_static") == 0 || strcmp(verb, "cgo_export_dynamic") == 0) {
		local = getimpsym(&q);
		if(local == nil)
			goto err2;
		if(!more(&q)) {
			fmtprint(&pragcgobuf, "%s %q\n", verb, local);
			goto out;
		}
		remote = getimpsym(&q);
		if(remote == nil)
			goto err2;
		fmtprint(&pragcgobuf, "%s %q %q\n", verb, local, remote);
		goto out;
	
	err2:
		yyerror("usage: //go:%s local [remote]", verb);
		goto out;
	}
	
	if(strcmp(verb, "cgo_import_dynamic") == 0 || strcmp(verb, "dynimport") == 0) {
		local = getimpsym(&q);
		if(local == nil)
			goto err3;
		if(!more(&q)) {
			fmtprint(&pragcgobuf, "cgo_import_dynamic %q\n", local);
			goto out;
		}
		remote = getimpsym(&q);
		if(remote == nil)
			goto err3;
		if(!more(&q)) {
			fmtprint(&pragcgobuf, "cgo_import_dynamic %q %q\n", local, remote);
			goto out;
		}
		p = getquoted(&q);
		if(p == nil)	
			goto err3;
		fmtprint(&pragcgobuf, "cgo_import_dynamic %q %q %q\n", local, remote, p);
		goto out;
	
	err3:
		yyerror("usage: //go:cgo_import_dynamic local [remote [\"library\"]]");
		goto out;
	}
	
	if(strcmp(verb, "cgo_import_static") == 0) {
		local = getimpsym(&q);
		if(local == nil || more(&q))
			goto err4;
		fmtprint(&pragcgobuf, "cgo_import_static %q\n", local);
		goto out;
		
	err4:
		yyerror("usage: //go:cgo_import_static local");
		goto out;
	}
	
	if(strcmp(verb, "cgo_ldflag") == 0) {
		p = getquoted(&q);
		if(p == nil)
			goto err5;
		fmtprint(&pragcgobuf, "cgo_ldflag %q\n", p);
		goto out;

	err5:
		yyerror("usage: //go:cgo_ldflag \"arg\"");
		goto out;
	}

out:;
}

int32
yylex(void)
{
	int lx;
	
	lx = _yylex();
	
	if(curio.nlsemi && lx == EOF) {
		// Treat EOF as "end of line" for the purposes
		// of inserting a semicolon.
		lx = ';';
	}

	switch(lx) {
	case LNAME:
	case LLITERAL:
	case LBREAK:
	case LCONTINUE:
	case LFALL:
	case LRETURN:
	case LINC:
	case LDEC:
	case ')':
	case '}':
	case ']':
		curio.nlsemi = 1;
		break;
	default:
		curio.nlsemi = 0;
		break;
	}

	// Track last two tokens returned by yylex.
	yyprev = yylast;
	yylast = lx;
	return lx;
}

static int
getc(void)
{
	int c, c1, c2;

	c = curio.peekc;
	if(c != 0) {
		curio.peekc = curio.peekc1;
		curio.peekc1 = 0;
		goto check;
	}
	
	if(curio.bin == nil) {
		c = *curio.cp & 0xff;
		if(c != 0)
			curio.cp++;
	} else {
	loop:
		c = BGETC(curio.bin);
		if(c == 0xef) {
			c1 = BGETC(curio.bin);
			c2 = BGETC(curio.bin);
			if(c1 == 0xbb && c2 == 0xbf) {
				yyerrorl(lexlineno, "Unicode (UTF-8) BOM in middle of file");
				goto loop;
			}
			Bungetc(curio.bin);
			Bungetc(curio.bin);
		}
	}

check:
	switch(c) {
	case 0:
		if(curio.bin != nil) {
			yyerror("illegal NUL byte");
			break;
		}
	case EOF:
		// insert \n at EOF
		if(curio.eofnl || curio.last == '\n')
			return EOF;
		curio.eofnl = 1;
		c = '\n';
	case '\n':
		if(pushedio.bin == nil)
			lexlineno++;
		break;
	}
	curio.last = c;
	return c;
}

static void
ungetc(int c)
{
	curio.peekc1 = curio.peekc;
	curio.peekc = c;
	if(c == '\n' && pushedio.bin == nil)
		lexlineno--;
}

static int32
getr(void)
{
	int c, i;
	char str[UTFmax+1];
	Rune rune;

	c = getc();
	if(c < Runeself)
		return c;
	i = 0;
	str[i++] = c;

loop:
	c = getc();
	str[i++] = c;
	if(!fullrune(str, i))
		goto loop;
	c = chartorune(&rune, str);
	if(rune == Runeerror && c == 1) {
		lineno = lexlineno;
		yyerror("illegal UTF-8 sequence");
		flusherrors();
		print("\t");
		for(c=0; c<i; c++)
			print("%s%.2x", c > 0 ? " " : "", *(uchar*)(str+c));
		print("\n");
	}
	return rune;
}

static int
escchar(int e, int *escflg, vlong *val)
{
	int i, u, c;
	vlong l;

	*escflg = 0;

	c = getr();
	switch(c) {
	case EOF:
		yyerror("eof in string");
		return 1;
	case '\n':
		yyerror("newline in string");
		return 1;
	case '\\':
		break;
	default:
		if(c == e)
			return 1;
		*val = c;
		return 0;
	}

	u = 0;
	c = getr();
	switch(c) {
	case 'x':
		*escflg = 1;	// it's a byte
		i = 2;
		goto hex;

	case 'u':
		i = 4;
		u = 1;
		goto hex;

	case 'U':
		i = 8;
		u = 1;
		goto hex;

	case '0':
	case '1':
	case '2':
	case '3':
	case '4':
	case '5':
	case '6':
	case '7':
		*escflg = 1;	// it's a byte
		goto oct;

	case 'a': c = '\a'; break;
	case 'b': c = '\b'; break;
	case 'f': c = '\f'; break;
	case 'n': c = '\n'; break;
	case 'r': c = '\r'; break;
	case 't': c = '\t'; break;
	case 'v': c = '\v'; break;
	case '\\': c = '\\'; break;

	default:
		if(c != e)
			yyerror("unknown escape sequence: %c", c);
	}
	*val = c;
	return 0;

hex:
	l = 0;
	for(; i>0; i--) {
		c = getc();
		if(c >= '0' && c <= '9') {
			l = l*16 + c-'0';
			continue;
		}
		if(c >= 'a' && c <= 'f') {
			l = l*16 + c-'a' + 10;
			continue;
		}
		if(c >= 'A' && c <= 'F') {
			l = l*16 + c-'A' + 10;
			continue;
		}
		yyerror("non-hex character in escape sequence: %c", c);
		ungetc(c);
		break;
	}
	if(u && (l > Runemax || (0xd800 <= l && l < 0xe000))) {
		yyerror("invalid Unicode code point in escape sequence: %#llx", l);
		l = Runeerror;
	}
	*val = l;
	return 0;

oct:
	l = c - '0';
	for(i=2; i>0; i--) {
		c = getc();
		if(c >= '0' && c <= '7') {
			l = l*8 + c-'0';
			continue;
		}
		yyerror("non-octal character in escape sequence: %c", c);
		ungetc(c);
	}
	if(l > 255)
		yyerror("octal escape value > 255: %d", l);

	*val = l;
	return 0;
}

static	struct
{
	char*	name;
	int	lexical;
	int	etype;
	int	op;
} syms[] =
{
/*	name		lexical		etype		op
 */
/* basic types */
	{"int8",		LNAME,		TINT8,		OXXX},
	{"int16",	LNAME,		TINT16,		OXXX},
	{"int32",	LNAME,		TINT32,		OXXX},
	{"int64",	LNAME,		TINT64,		OXXX},

	{"uint8",	LNAME,		TUINT8,		OXXX},
	{"uint16",	LNAME,		TUINT16,	OXXX},
	{"uint32",	LNAME,		TUINT32,	OXXX},
	{"uint64",	LNAME,		TUINT64,	OXXX},

	{"float32",	LNAME,		TFLOAT32,	OXXX},
	{"float64",	LNAME,		TFLOAT64,	OXXX},

	{"complex64",	LNAME,		TCOMPLEX64,	OXXX},
	{"complex128",	LNAME,		TCOMPLEX128,	OXXX},

	{"bool",		LNAME,		TBOOL,		OXXX},
	{"string",	LNAME,		TSTRING,	OXXX},

	{"any",		LNAME,		TANY,		OXXX},

	{"break",	LBREAK,		Txxx,		OXXX},
	{"case",		LCASE,		Txxx,		OXXX},
	{"chan",		LCHAN,		Txxx,		OXXX},
	{"const",	LCONST,		Txxx,		OXXX},
	{"continue",	LCONTINUE,	Txxx,		OXXX},
	{"default",	LDEFAULT,	Txxx,		OXXX},
	{"else",		LELSE,		Txxx,		OXXX},
	{"defer",	LDEFER,		Txxx,		OXXX},
	{"fallthrough",	LFALL,		Txxx,		OXXX},
	{"for",		LFOR,		Txxx,		OXXX},
	{"func",		LFUNC,		Txxx,		OXXX},
	{"go",		LGO,		Txxx,		OXXX},
	{"goto",		LGOTO,		Txxx,		OXXX},
	{"if",		LIF,		Txxx,		OXXX},
	{"import",	LIMPORT,	Txxx,		OXXX},
	{"interface",	LINTERFACE,	Txxx,		OXXX},
	{"map",		LMAP,		Txxx,		OXXX},
	{"package",	LPACKAGE,	Txxx,		OXXX},
	{"range",	LRANGE,		Txxx,		OXXX},
	{"return",	LRETURN,	Txxx,		OXXX},
	{"select",	LSELECT,	Txxx,		OXXX},
	{"struct",	LSTRUCT,	Txxx,		OXXX},
	{"switch",	LSWITCH,	Txxx,		OXXX},
	{"type",		LTYPE,		Txxx,		OXXX},
	{"var",		LVAR,		Txxx,		OXXX},

	{"append",	LNAME,		Txxx,		OAPPEND},
	{"cap",		LNAME,		Txxx,		OCAP},
	{"close",	LNAME,		Txxx,		OCLOSE},
	{"complex",	LNAME,		Txxx,		OCOMPLEX},
	{"copy",		LNAME,		Txxx,		OCOPY},
	{"delete",	LNAME,		Txxx,		ODELETE},
	{"imag",		LNAME,		Txxx,		OIMAG},
	{"len",		LNAME,		Txxx,		OLEN},
	{"make",		LNAME,		Txxx,		OMAKE},
	{"new",		LNAME,		Txxx,		ONEW},
	{"panic",	LNAME,		Txxx,		OPANIC},
	{"print",	LNAME,		Txxx,		OPRINT},
	{"println",	LNAME,		Txxx,		OPRINTN},
	{"real",		LNAME,		Txxx,		OREAL},
	{"recover",	LNAME,		Txxx,		ORECOVER},

	{"notwithstanding",		LIGNORE,	Txxx,		OXXX},
	{"thetruthofthematter",		LIGNORE,	Txxx,		OXXX},
	{"despiteallobjections",		LIGNORE,	Txxx,		OXXX},
	{"whereas",			LIGNORE,	Txxx,		OXXX},
	{"insofaras",			LIGNORE,	Txxx,		OXXX},
};

static void
lexinit(void)
{
	int i, lex;
	Sym *s, *s1;
	Type *t;
	int etype;
	Val v;

	/*
	 * initialize basic types array
	 * initialize known symbols
	 */
	for(i=0; i<nelem(syms); i++) {
		lex = syms[i].lexical;
		s = lookup(syms[i].name);
		s->lexical = lex;

		etype = syms[i].etype;
		if(etype != Txxx) {
			if(etype < 0 || etype >= nelem(types))
				fatal("lexinit: %s bad etype", s->name);
			s1 = pkglookup(syms[i].name, builtinpkg);
			t = types[etype];
			if(t == T) {
				t = typ(etype);
				t->sym = s1;

				if(etype != TANY && etype != TSTRING)
					dowidth(t);
				types[etype] = t;
			}
			s1->lexical = LNAME;
			s1->def = typenod(t);
			continue;
		}

		etype = syms[i].op;
		if(etype != OXXX) {
			s1 = pkglookup(syms[i].name, builtinpkg);
			s1->lexical = LNAME;
			s1->def = nod(ONAME, N, N);
			s1->def->sym = s1;
			s1->def->etype = etype;
			s1->def->builtin = 1;
		}
	}

	// logically, the type of a string literal.
	// types[TSTRING] is the named type string
	// (the type of x in var x string or var x = "hello").
	// this is the ideal form
	// (the type of x in const x = "hello").
	idealstring = typ(TSTRING);
	idealbool = typ(TBOOL);

	s = pkglookup("true", builtinpkg);
	s->def = nodbool(1);
	s->def->sym = lookup("true");
	s->def->type = idealbool;

	s = pkglookup("false", builtinpkg);
	s->def = nodbool(0);
	s->def->sym = lookup("false");
	s->def->type = idealbool;

	s = lookup("_");
	s->block = -100;
	s->def = nod(ONAME, N, N);
	s->def->sym = s;
	types[TBLANK] = typ(TBLANK);
	s->def->type = types[TBLANK];
	nblank = s->def;

	s = pkglookup("_", builtinpkg);
	s->block = -100;
	s->def = nod(ONAME, N, N);
	s->def->sym = s;
	types[TBLANK] = typ(TBLANK);
	s->def->type = types[TBLANK];

	types[TNIL] = typ(TNIL);
	s = pkglookup("nil", builtinpkg);
	v.ctype = CTNIL;
	s->def = nodlit(v);
	s->def->sym = s;
}

static void
lexinit1(void)
{
	Sym *s, *s1;
	Type *t, *f, *rcvr, *in, *out;

	// t = interface { Error() string }
	rcvr = typ(TSTRUCT);
	rcvr->type = typ(TFIELD);
	rcvr->type->type = ptrto(typ(TSTRUCT));
	rcvr->funarg = 1;
	in = typ(TSTRUCT);
	in->funarg = 1;
	out = typ(TSTRUCT);
	out->type = typ(TFIELD);
	out->type->type = types[TSTRING];
	out->funarg = 1;
	f = typ(TFUNC);
	*getthis(f) = rcvr;
	*getoutarg(f) = out;
	*getinarg(f) = in;
	f->thistuple = 1;
	f->intuple = 0;
	f->outnamed = 0;
	f->outtuple = 1;
	t = typ(TINTER);
	t->type = typ(TFIELD);
	t->type->sym = lookup("Error");
	t->type->type = f;

	// error type
	s = lookup("error");
	s->lexical = LNAME;
	s1 = pkglookup("error", builtinpkg);
	errortype = t;
	errortype->sym = s1;
	s1->lexical = LNAME;
	s1->def = typenod(errortype);

	// byte alias
	s = lookup("byte");
	s->lexical = LNAME;
	s1 = pkglookup("byte", builtinpkg);
	bytetype = typ(TUINT8);
	bytetype->sym = s1;
	s1->lexical = LNAME;
	s1->def = typenod(bytetype);

	// rune alias
	s = lookup("rune");
	s->lexical = LNAME;
	s1 = pkglookup("rune", builtinpkg);
	runetype = typ(TINT32);
	runetype->sym = s1;
	s1->lexical = LNAME;
	s1->def = typenod(runetype);
}

static void
lexfini(void)
{
	Sym *s;
	int lex, etype, i;
	Val v;

	for(i=0; i<nelem(syms); i++) {
		lex = syms[i].lexical;
		if(lex != LNAME)
			continue;
		s = lookup(syms[i].name);
		s->lexical = lex;

		etype = syms[i].etype;
		if(etype != Txxx && (etype != TANY || debug['A']) && s->def == N) {
			s->def = typenod(types[etype]);
			s->origpkg = builtinpkg;
		}

		etype = syms[i].op;
		if(etype != OXXX && s->def == N) {
			s->def = nod(ONAME, N, N);
			s->def->sym = s;
			s->def->etype = etype;
			s->def->builtin = 1;
			s->origpkg = builtinpkg;
		}
	}

	// backend-specific builtin types (e.g. int).
	for(i=0; arch.typedefs[i].name; i++) {
		s = lookup(arch.typedefs[i].name);
		if(s->def == N) {
			s->def = typenod(types[arch.typedefs[i].etype]);
			s->origpkg = builtinpkg;
		}
	}

	// there's only so much table-driven we can handle.
	// these are special cases.
	s = lookup("byte");
	if(s->def == N) {
		s->def = typenod(bytetype);
		s->origpkg = builtinpkg;
	}

	s = lookup("error");
	if(s->def == N) {
		s->def = typenod(errortype);
		s->origpkg = builtinpkg;
	}

	s = lookup("rune");
	if(s->def == N) {
		s->def = typenod(runetype);
		s->origpkg = builtinpkg;
	}

	s = lookup("nil");
	if(s->def == N) {
		v.ctype = CTNIL;
		s->def = nodlit(v);
		s->def->sym = s;
		s->origpkg = builtinpkg;
	}

	s = lookup("iota");
	if(s->def == N) {
		s->def = nod(OIOTA, N, N);
		s->def->sym = s;
		s->origpkg = builtinpkg;
	}

	s = lookup("true");
	if(s->def == N) {
		s->def = nodbool(1);
		s->def->sym = s;
		s->origpkg = builtinpkg;
	}

	s = lookup("false");
	if(s->def == N) {
		s->def = nodbool(0);
		s->def->sym = s;
		s->origpkg = builtinpkg;
	}

	nodfp = nod(ONAME, N, N);
	nodfp->type = types[TINT32];
	nodfp->xoffset = 0;
	nodfp->class = PPARAM;
	nodfp->sym = lookup(".fp");
}

struct
{
	int	lex;
	char*	name;
} lexn[] =
{
	{LANDAND,	"ANDAND"},
	{LANDNOT,	"ANDNOT"},
	{LASOP,		"ASOP"},
	{LBREAK,		"BREAK"},
	{LCASE,		"CASE"},
	{LCHAN,		"CHAN"},
	{LCOLAS,		"COLAS"},
	{LCOMM,		"<-"},
	{LCONST,		"CONST"},
	{LCONTINUE,	"CONTINUE"},
	{LDDD,		"..."},
	{LDEC,		"DEC"},
	{LDEFAULT,	"DEFAULT"},
	{LDEFER,		"DEFER"},
	{LELSE,		"ELSE"},
	{LEQ,		"EQ"},
	{LFALL,		"FALL"},
	{LFOR,		"FOR"},
	{LFUNC,		"FUNC"},
	{LGE,		"GE"},
	{LGO,		"GO"},
	{LGOTO,		"GOTO"},
	{LGT,		"GT"},
	{LIF,		"IF"},
	{LIMPORT,	"IMPORT"},
	{LINC,		"INC"},
	{LINTERFACE,	"INTERFACE"},
	{LLE,		"LE"},
	{LLITERAL,	"LITERAL"},
	{LLSH,		"LSH"},
	{LLT,		"LT"},
	{LMAP,		"MAP"},
	{LNAME,		"NAME"},
	{LNE,		"NE"},
	{LOROR,		"OROR"},
	{LPACKAGE,	"PACKAGE"},
	{LRANGE,		"RANGE"},
	{LRETURN,	"RETURN"},
	{LRSH,		"RSH"},
	{LSELECT,	"SELECT"},
	{LSTRUCT,	"STRUCT"},
	{LSWITCH,	"SWITCH"},
	{LTYPE,		"TYPE"},
	{LVAR,		"VAR"},
};

char*
lexname(int lex)
{
	int i;
	static char buf[100];

	for(i=0; i<nelem(lexn); i++)
		if(lexn[i].lex == lex)
			return lexn[i].name;
	snprint(buf, sizeof(buf), "LEX-%d", lex);
	return buf;
}

struct
{
	char *have;
	char *want;
} yytfix[] =
{
	{"$end",	"EOF"},
	{"LLITERAL",	"literal"},
	{"LASOP",	"op="},
	{"LBREAK",	"break"},
	{"LCASE",	"case"},
	{"LCHAN",	"chan"},
	{"LCOLAS",	":="},
	{"LCONST",	"const"},
	{"LCONTINUE",	"continue"},
	{"LDDD",	"..."},
	{"LDEFAULT",	"default"},
	{"LDEFER",	"defer"},
	{"LELSE",	"else"},
	{"LFALL",	"fallthrough"},
	{"LFOR",	"for"},
	{"LFUNC",	"func"},
	{"LGO",	"go"},
	{"LGOTO",	"goto"},
	{"LIF",	"if"},
	{"LIMPORT",	"import"},
	{"LINTERFACE",	"interface"},
	{"LMAP",	"map"},
	{"LNAME",	"name"},
	{"LPACKAGE",	"package"},
	{"LRANGE",	"range"},
	{"LRETURN",	"return"},
	{"LSELECT",	"select"},
	{"LSTRUCT",	"struct"},
	{"LSWITCH",	"switch"},
	{"LTYPE",	"type"},
	{"LVAR",	"var"},
	{"LANDAND",	"&&"},
	{"LANDNOT",	"&^"},
	{"LBODY",	"{"},
	{"LCOMM",	"<-"},
	{"LDEC",	"--"},
	{"LINC",	"++"},
	{"LEQ",	"=="},
	{"LGE",	">="},
	{"LGT",	">"},
	{"LLE",	"<="},
	{"LLT",	"<"},
	{"LLSH",	"<<"},
	{"LRSH",	">>"},
	{"LOROR",	"||"},
	{"LNE",	"!="},
	
	// spell out to avoid confusion with punctuation in error messages
	{"';'",	"semicolon or newline"},
	{"','",	"comma"},
};

static void
yytinit(void)
{
	int i, j;
	extern char *yytname[];
	char *s, *t;

	for(i=0; yytname[i] != nil; i++) {
		s = yytname[i];
		
		if(strcmp(s, "LLITERAL") == 0) {
			strcpy(litbuf, "literal");
			yytname[i] = litbuf;
			goto loop;
		}
		
		// apply yytfix if possible
		for(j=0; j<nelem(yytfix); j++) {
			if(strcmp(s, yytfix[j].have) == 0) {
				yytname[i] = yytfix[j].want;
				goto loop;
			}
		}

		// turn 'x' into x.
		if(s[0] == '\'') {
			t = strdup(s+1);
			t[strlen(t)-1] = '\0';
			yytname[i] = t;
		}
	loop:;
	}		
}

static void
pkgnotused(int lineno, Strlit *path, char *name)
{
	char *elem;
	
	// If the package was imported with a name other than the final
	// import path element, show it explicitly in the error message.
	// Note that this handles both renamed imports and imports of
	// packages containing unconventional package declarations.
	// Note that this uses / always, even on Windows, because Go import
	// paths always use forward slashes.
	elem = strrchr(path->s, '/');
	if(elem != nil)
		elem++;
	else
		elem = path->s;
	if(name == nil || strcmp(elem, name) == 0)
		yyerrorl(lineno, "imported and not used: \"%Z\"", path);
	else
		yyerrorl(lineno, "imported and not used: \"%Z\" as %s", path, name);
}

void
mkpackage(char* pkgname)
{
	Sym *s;
	int32 h;
	char *p, *q;

	if(localpkg->name == nil) {
		if(strcmp(pkgname, "_") == 0)
			yyerror("invalid package name _");
		localpkg->name = pkgname;
	} else {
		if(strcmp(pkgname, localpkg->name) != 0)
			yyerror("package %s; expected %s", pkgname, localpkg->name);
		for(h=0; h<NHASH; h++) {
			for(s = hash[h]; s != S; s = s->link) {
				if(s->def == N || s->pkg != localpkg)
					continue;
				if(s->def->op == OPACK) {
					// throw away top-level package name leftover
					// from previous file.
					// leave s->block set to cause redeclaration
					// errors if a conflicting top-level name is
					// introduced by a different file.
					if(!s->def->used && !nsyntaxerrors)
						pkgnotused(s->def->lineno, s->def->pkg->path, s->name);
					s->def = N;
					continue;
				}
				if(s->def->sym != s) {
					// throw away top-level name left over
					// from previous import . "x"
					if(s->def->pack != N && !s->def->pack->used && !nsyntaxerrors) {
						pkgnotused(s->def->pack->lineno, s->def->pack->pkg->path, nil);
						s->def->pack->used = 1;
					}
					s->def = N;
					continue;
				}
			}
		}
	}

	if(outfile == nil) {
		p = strrchr(infile, '/');
		if(ctxt->windows) {
			q = strrchr(infile, '\\');
			if(q > p)
				p = q;
		}
		if(p == nil)
			p = infile;
		else
			p = p+1;
		snprint(namebuf, sizeof(namebuf), "%s", p);
		p = strrchr(namebuf, '.');
		if(p != nil)
			*p = 0;
		outfile = smprint("%s.%c", namebuf, arch.thechar);
	}
}
