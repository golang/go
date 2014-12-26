// Derived from Inferno utils/6l/obj.c and utils/6l/span.c
// http://code.google.com/p/inferno-os/source/browse/utils/6l/obj.c
// http://code.google.com/p/inferno-os/source/browse/utils/6l/span.c
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

#include	"l.h"
#include	"lib.h"
#include	"../ld/elf.h"
#include	"../ld/dwarf.h"
#include	"../../runtime/stack.h"
#include	"../../runtime/funcdata.h"

#include	<ar.h>
#if !(defined(_WIN32) || defined(PLAN9))
#include	<sys/stat.h>
#endif

enum
{
	// Whether to assume that the external linker is "gold"
	// (http://sourceware.org/ml/binutils/2008-03/msg00162.html).
	AssumeGoldLinker = 0,
};

int iconv(Fmt*);

char	symname[]	= SYMDEF;
char	pkgname[]	= "__.PKGDEF";
static int	cout = -1;

extern int	version;

// Set if we see an object compiled by the host compiler that is not
// from a package that is known to support internal linking mode.
static int	externalobj = 0;

static	void	hostlinksetup(void);

char*	goroot;
char*	goarch;
char*	goos;
char*	theline;

void
Lflag(char *arg)
{
	char **p;

	if(ctxt->nlibdir >= ctxt->maxlibdir) {
		if (ctxt->maxlibdir == 0)
			ctxt->maxlibdir = 8;
		else
			ctxt->maxlibdir *= 2;
		p = erealloc(ctxt->libdir, ctxt->maxlibdir * sizeof(*p));
		ctxt->libdir = p;
	}
	ctxt->libdir[ctxt->nlibdir++] = arg;
}

/*
 * Unix doesn't like it when we write to a running (or, sometimes,
 * recently run) binary, so remove the output file before writing it.
 * On Windows 7, remove() can force a subsequent create() to fail.
 * S_ISREG() does not exist on Plan 9.
 */
static void
mayberemoveoutfile(void) 
{
#if !(defined(_WIN32) || defined(PLAN9))
	struct stat st;
	if(lstat(outfile, &st) == 0 && !S_ISREG(st.st_mode))
		return;
#endif
	remove(outfile);
}

void
libinit(void)
{
	char *suffix, *suffixsep;

	funcalign = FuncAlign;
	fmtinstall('i', iconv);
	fmtinstall('Y', Yconv);
	fmtinstall('Z', Zconv);
	mywhatsys();	// get goroot, goarch, goos

	// add goroot to the end of the libdir list.
	suffix = "";
	suffixsep = "";
	if(flag_installsuffix != nil) {
		suffixsep = "_";
		suffix = flag_installsuffix;
	} else if(flag_race) {
		suffixsep = "_";
		suffix = "race";
	}
	Lflag(smprint("%s/pkg/%s_%s%s%s", goroot, goos, goarch, suffixsep, suffix));

	mayberemoveoutfile();
	cout = create(outfile, 1, 0775);
	if(cout < 0) {
		diag("cannot create %s: %r", outfile);
		errorexit();
	}

	if(INITENTRY == nil) {
		INITENTRY = mal(strlen(goarch)+strlen(goos)+20);
		if(!flag_shared) {
			sprint(INITENTRY, "_rt0_%s_%s", goarch, goos);
		} else {
			sprint(INITENTRY, "_rt0_%s_%s_lib", goarch, goos);
		}
	}
	linklookup(ctxt, INITENTRY, 0)->type = SXREF;
}

void
errorexit(void)
{
	if(cout >= 0) {
		// For rmtemp run at atexit time on Windows.
		close(cout);
	}
	if(nerrors) {
		if(cout >= 0)
			mayberemoveoutfile();
		exits("error");
	}
	exits(0);
}

void
loadinternal(char *name)
{
	char pname[1024];
	int i, found;

	found = 0;
	for(i=0; i<ctxt->nlibdir; i++) {
		snprint(pname, sizeof pname, "%s/%s.a", ctxt->libdir[i], name);
		if(debug['v'])
			Bprint(&bso, "searching for %s.a in %s\n", name, pname);
		if(access(pname, AEXIST) >= 0) {
			addlibpath(ctxt, "internal", "internal", pname, name);
			found = 1;
			break;
		}
	}
	if(!found)
		Bprint(&bso, "warning: unable to find %s.a\n", name);
}

void
loadlib(void)
{
	int i, w, x;
	LSym *s, *tlsg;
	char* cgostrsym;

	if(flag_shared) {
		s = linklookup(ctxt, "runtime.islibrary", 0);
		s->dupok = 1;
		adduint8(ctxt, s, 1);
	}

	loadinternal("runtime");
	if(thechar == '5')
		loadinternal("math");
	if(flag_race)
		loadinternal("runtime/race");

	for(i=0; i<ctxt->libraryp; i++) {
		if(debug['v'] > 1)
			Bprint(&bso, "%5.2f autolib: %s (from %s)\n", cputime(), ctxt->library[i].file, ctxt->library[i].objref);
		iscgo |= strcmp(ctxt->library[i].pkg, "runtime/cgo") == 0;
		objfile(ctxt->library[i].file, ctxt->library[i].pkg);
	}

	if(linkmode == LinkAuto) {
		if(iscgo && externalobj)
			linkmode = LinkExternal;
		else
			linkmode = LinkInternal;

		// Force external linking for android.
		if(strcmp(goos, "android") == 0)
			linkmode = LinkExternal;

		// cgo on Darwin must use external linking
		// we can always use external linking, but then there will be circular
		// dependency problems when compiling natively (external linking requires
		// runtime/cgo, runtime/cgo requires cmd/cgo, but cmd/cgo needs to be
		// compiled using external linking.)
		if(thechar == '5' && HEADTYPE == Hdarwin && iscgo)
			linkmode = LinkExternal;
	}

	if(linkmode == LinkExternal && !iscgo) {
		// This indicates a user requested -linkmode=external.
		// The startup code uses an import of runtime/cgo to decide
		// whether to initialize the TLS.  So give it one.  This could
		// be handled differently but it's an unusual case.
		loadinternal("runtime/cgo");
		if(i < ctxt->libraryp)
			objfile(ctxt->library[i].file, ctxt->library[i].pkg);

		// Pretend that we really imported the package.
		s = linklookup(ctxt, "go.importpath.runtime/cgo.", 0);
		s->type = SDATA;
		s->dupok = 1;
		s->reachable = 1;

		// Provided by the code that imports the package.
		// Since we are simulating the import, we have to provide this string.
		cgostrsym = "go.string.\"runtime/cgo\"";
		if(linkrlookup(ctxt, cgostrsym, 0) == nil) {
			s = linklookup(ctxt, cgostrsym, 0);
			s->type = SRODATA;
			s->reachable = 1;
			addstrdata(cgostrsym, "runtime/cgo");
		}
	}

	if(linkmode == LinkInternal) {
		// Drop all the cgo_import_static declarations.
		// Turns out we won't be needing them.
		for(s = ctxt->allsym; s != S; s = s->allsym)
			if(s->type == SHOSTOBJ) {
				// If a symbol was marked both
				// cgo_import_static and cgo_import_dynamic,
				// then we want to make it cgo_import_dynamic
				// now.
				if(s->extname != nil && s->dynimplib != nil && s->cgoexport == 0) {
					s->type = SDYNIMPORT;
				} else
					s->type = 0;
			}
	}
	
	tlsg = linklookup(ctxt, "runtime.tlsg", 0);
	// For most ports, runtime.tlsg is a placeholder symbol for TLS
	// relocation. However, the Android and Darwin ports need it to
	// be a real variable. Instead of hard-coding which platforms
	// need it to be a real variable, we set the type to STLSBSS only
	// when the runtime has not declared its type already.
	if(tlsg->type == 0)
		tlsg->type = STLSBSS;
	tlsg->size = PtrSize;
	tlsg->hide = 1;
	tlsg->reachable = 1;
	ctxt->tlsg = tlsg;

	// Now that we know the link mode, trim the dynexp list.
	x = CgoExportDynamic;
	if(linkmode == LinkExternal)
		x = CgoExportStatic;
	w = 0;
	for(i=0; i<ndynexp; i++)
		if(dynexp[i]->cgoexport & x)
			dynexp[w++] = dynexp[i];
	ndynexp = w;
	
	// In internal link mode, read the host object files.
	if(linkmode == LinkInternal)
		hostobjs();
	else
		hostlinksetup();

	// We've loaded all the code now.
	// If there are no dynamic libraries needed, gcc disables dynamic linking.
	// Because of this, glibc's dynamic ELF loader occasionally (like in version 2.13)
	// assumes that a dynamic binary always refers to at least one dynamic library.
	// Rather than be a source of test cases for glibc, disable dynamic linking
	// the same way that gcc would.
	//
	// Exception: on OS X, programs such as Shark only work with dynamic
	// binaries, so leave it enabled on OS X (Mach-O) binaries.
	// Also leave it enabled on Solaris which doesn't support
	// statically linked binaries.
	if(!flag_shared && !havedynamic && HEADTYPE != Hdarwin && HEADTYPE != Hsolaris)
		debug['d'] = 1;
	
	importcycles();
}

/*
 * look for the next file in an archive.
 * adapted from libmach.
 */
static vlong
nextar(Biobuf *bp, vlong off, struct ar_hdr *a)
{
	int r;
	int32 arsize;
	char *buf;

	if (off&01)
		off++;
	Bseek(bp, off, 0);
	buf = Brdline(bp, '\n');
	r = Blinelen(bp);
	if(buf == nil) {
		if(r == 0)
			return 0;
		return -1;
	}
	if(r != SAR_HDR)
		return -1;
	memmove(a, buf, SAR_HDR);
	if(strncmp(a->fmag, ARFMAG, sizeof a->fmag))
		return -1;
	arsize = strtol(a->size, 0, 0);
	if (arsize&1)
		arsize++;
	return arsize + r;
}

void
objfile(char *file, char *pkg)
{
	vlong off, l;
	Biobuf *f;
	char magbuf[SARMAG];
	char pname[150];
	struct ar_hdr arhdr;

	pkg = smprint("%i", pkg);

	if(debug['v'] > 1)
		Bprint(&bso, "%5.2f ldobj: %s (%s)\n", cputime(), file, pkg);
	Bflush(&bso);
	f = Bopen(file, 0);
	if(f == nil) {
		diag("cannot open file: %s", file);
		errorexit();
	}
	l = Bread(f, magbuf, SARMAG);
	if(l != SARMAG || strncmp(magbuf, ARMAG, SARMAG)){
		/* load it as a regular file */
		l = Bseek(f, 0L, 2);
		Bseek(f, 0L, 0);
		ldobj(f, pkg, l, file, file, FileObj);
		Bterm(f);
		free(pkg);
		return;
	}
	
	/* skip over optional __.GOSYMDEF and process __.PKGDEF */
	off = Boffset(f);
	l = nextar(f, off, &arhdr);
	if(l <= 0) {
		diag("%s: short read on archive file symbol header", file);
		goto out;
	}
	if(strncmp(arhdr.name, symname, strlen(symname)) == 0) {
		off += l;
		l = nextar(f, off, &arhdr);
		if(l <= 0) {
			diag("%s: short read on archive file symbol header", file);
			goto out;
		}
	}

	if(strncmp(arhdr.name, pkgname, strlen(pkgname))) {
		diag("%s: cannot find package header", file);
		goto out;
	}
	off += l;

	if(debug['u'])
		ldpkg(f, pkg, atolwhex(arhdr.size), file, Pkgdef);

	/*
	 * load all the object files from the archive now.
	 * this gives us sequential file access and keeps us
	 * from needing to come back later to pick up more
	 * objects.  it breaks the usual C archive model, but
	 * this is Go, not C.  the common case in Go is that
	 * we need to load all the objects, and then we throw away
	 * the individual symbols that are unused.
	 *
	 * loading every object will also make it possible to
	 * load foreign objects not referenced by __.GOSYMDEF.
	 */
	for(;;) {
		l = nextar(f, off, &arhdr);
		if(l == 0)
			break;
		if(l < 0) {
			diag("%s: malformed archive", file);
			goto out;
		}
		off += l;

		l = SARNAME;
		while(l > 0 && arhdr.name[l-1] == ' ')
			l--;
		snprint(pname, sizeof pname, "%s(%.*s)", file, utfnlen(arhdr.name, l), arhdr.name);
		l = atolwhex(arhdr.size);
		ldobj(f, pkg, l, pname, file, ArchiveObj);
	}

out:
	Bterm(f);
	free(pkg);
}

static void
dowrite(int fd, char *p, int n)
{
	int m;
	
	while(n > 0) {
		m = write(fd, p, n);
		if(m <= 0) {
			ctxt->cursym = S;
			diag("write error: %r");
			errorexit();
		}
		n -= m;
		p += m;
	}
}

typedef struct Hostobj Hostobj;

struct Hostobj
{
	void (*ld)(Biobuf*, char*, int64, char*);
	char *pkg;
	char *pn;
	char *file;
	int64 off;
	int64 len;
};

Hostobj *hostobj;
int nhostobj;
int mhostobj;

// These packages can use internal linking mode.
// Others trigger external mode.
const char *internalpkg[] = {
	"crypto/x509",
	"net",
	"os/user",
	"runtime/cgo",
	"runtime/race"
};

void
ldhostobj(void (*ld)(Biobuf*, char*, int64, char*), Biobuf *f, char *pkg, int64 len, char *pn, char *file)
{
	int i, isinternal;
	Hostobj *h;

	isinternal = 0;
	for(i=0; i<nelem(internalpkg); i++) {
		if(strcmp(pkg, internalpkg[i]) == 0) {
			isinternal = 1;
			break;
		}
	}

	// DragonFly declares errno with __thread, which results in a symbol
	// type of R_386_TLS_GD or R_X86_64_TLSGD. The Go linker does not
	// currently know how to handle TLS relocations, hence we have to
	// force external linking for any libraries that link in code that
	// uses errno. This can be removed if the Go linker ever supports
	// these relocation types.
	if(HEADTYPE == Hdragonfly)
	if(strcmp(pkg, "net") == 0 || strcmp(pkg, "os/user") == 0)
		isinternal = 0;

	if(!isinternal)
		externalobj = 1;

	if(nhostobj >= mhostobj) {
		if(mhostobj == 0)
			mhostobj = 16;
		else
			mhostobj *= 2;
		hostobj = erealloc(hostobj, mhostobj*sizeof hostobj[0]);
	}
	h = &hostobj[nhostobj++];
	h->ld = ld;
	h->pkg = estrdup(pkg);
	h->pn = estrdup(pn);
	h->file = estrdup(file);
	h->off = Boffset(f);
	h->len = len;
}

void
hostobjs(void)
{
	int i;
	Biobuf *f;
	Hostobj *h;
	
	for(i=0; i<nhostobj; i++) {
		h = &hostobj[i];
		f = Bopen(h->file, OREAD);
		if(f == nil) {
			ctxt->cursym = S;
			diag("cannot reopen %s: %r", h->pn);
			errorexit();
		}
		Bseek(f, h->off, 0);
		h->ld(f, h->pkg, h->len, h->pn);
		Bterm(f);
	}
}

// provided by lib9
int runcmd(char**);
char* mktempdir(void);
void removeall(char*);

static void
rmtemp(void)
{
	removeall(tmpdir);
}

static void
hostlinksetup(void)
{
	char *p;

	if(linkmode != LinkExternal)
		return;

	// create temporary directory and arrange cleanup
	if(tmpdir == nil) {
		tmpdir = mktempdir();
		atexit(rmtemp);
	}

	// change our output to temporary object file
	close(cout);
	p = smprint("%s/go.o", tmpdir);
	cout = create(p, 1, 0775);
	if(cout < 0) {
		diag("cannot create %s: %r", p);
		errorexit();
	}
	free(p);
}

void
hostlink(void)
{
	char *p, **argv;
	int c, i, w, n, argc, len;
	Hostobj *h;
	Biobuf *f;
	static char buf[64<<10];

	if(linkmode != LinkExternal || nerrors > 0)
		return;

	c = 0;
	p = extldflags;
	while(p != nil) {
		while(*p == ' ')
			p++;
		if(*p == '\0')
			break;
		c++;
		p = strchr(p + 1, ' ');
	}

	argv = malloc((14+nhostobj+nldflag+c)*sizeof argv[0]);
	argc = 0;
	if(extld == nil)
		extld = "gcc";
	argv[argc++] = extld;
	switch(thechar){
	case '8':
		argv[argc++] = "-m32";
		break;
	case '6':
	case '9':
		argv[argc++] = "-m64";
		break;
	case '5':
		argv[argc++] = "-marm";
		break;
	}
	if(!debug['s'] && !debug_s) {
		argv[argc++] = "-gdwarf-2"; 
	} else {
		argv[argc++] = "-s";
	}
	if(HEADTYPE == Hdarwin)
		argv[argc++] = "-Wl,-no_pie,-pagezero_size,4000000";
	if(HEADTYPE == Hopenbsd)
		argv[argc++] = "-Wl,-nopie";
	
	if(iself && AssumeGoldLinker)
		argv[argc++] = "-Wl,--rosegment";

	if(flag_shared) {
		argv[argc++] = "-Wl,-Bsymbolic";
		argv[argc++] = "-shared";
	}
	argv[argc++] = "-o";
	argv[argc++] = outfile;
	
	if(rpath)
		argv[argc++] = smprint("-Wl,-rpath,%s", rpath);

	// Force global symbols to be exported for dlopen, etc.
	if(iself)
		argv[argc++] = "-rdynamic";

	if(strstr(argv[0], "clang") != nil)
		argv[argc++] = "-Qunused-arguments";

	// already wrote main object file
	// copy host objects to temporary directory
	for(i=0; i<nhostobj; i++) {
		h = &hostobj[i];
		f = Bopen(h->file, OREAD);
		if(f == nil) {
			ctxt->cursym = S;
			diag("cannot reopen %s: %r", h->pn);
			errorexit();
		}
		Bseek(f, h->off, 0);
		p = smprint("%s/%06d.o", tmpdir, i);
		argv[argc++] = p;
		w = create(p, 1, 0775);
		if(w < 0) {
			ctxt->cursym = S;
			diag("cannot create %s: %r", p);
			errorexit();
		}
		len = h->len;
		while(len > 0 && (n = Bread(f, buf, sizeof buf)) > 0){
			if(n > len)
				n = len;
			dowrite(w, buf, n);
			len -= n;
		}
		if(close(w) < 0) {
			ctxt->cursym = S;
			diag("cannot write %s: %r", p);
			errorexit();
		}
		Bterm(f);
	}
	
	argv[argc++] = smprint("%s/go.o", tmpdir);
	for(i=0; i<nldflag; i++)
		argv[argc++] = ldflag[i];

	p = extldflags;
	while(p != nil) {
		while(*p == ' ')
			*p++ = '\0';
		if(*p == '\0')
			break;
		argv[argc++] = p;

		// clang, unlike GCC, passes -rdynamic to the linker
		// even when linking with -static, causing a linker
		// error when using GNU ld.  So take out -rdynamic if
		// we added it.  We do it in this order, rather than
		// only adding -rdynamic later, so that -extldflags
		// can override -rdynamic without using -static.
		if(iself && strncmp(p, "-static", 7) == 0 && (p[7]==' ' || p[7]=='\0')) {
			for(i=0; i<argc; i++) {
				if(strcmp(argv[i], "-rdynamic") == 0)
					argv[i] = "-static";
			}
		}

		p = strchr(p + 1, ' ');
	}

	argv[argc] = nil;

	quotefmtinstall();
	if(debug['v']) {
		Bprint(&bso, "host link:");
		for(i=0; i<argc; i++)
			Bprint(&bso, " %q", argv[i]);
		Bprint(&bso, "\n");
		Bflush(&bso);
	}

	if(runcmd(argv) < 0) {
		ctxt->cursym = S;
		diag("%s: running %s failed: %r", argv0, argv[0]);
		errorexit();
	}
}

void
ldobj(Biobuf *f, char *pkg, int64 len, char *pn, char *file, int whence)
{
	char *line;
	int n, c1, c2, c3, c4;
	uint32 magic;
	vlong import0, import1, eof;
	char *t;

	eof = Boffset(f) + len;

	pn = estrdup(pn);

	c1 = BGETC(f);
	c2 = BGETC(f);
	c3 = BGETC(f);
	c4 = BGETC(f);
	Bungetc(f);
	Bungetc(f);
	Bungetc(f);
	Bungetc(f);

	magic = c1<<24 | c2<<16 | c3<<8 | c4;
	if(magic == 0x7f454c46) {	// \x7F E L F
		ldhostobj(ldelf, f, pkg, len, pn, file);
		return;
	}
	if((magic&~1) == 0xfeedface || (magic&~0x01000000) == 0xcefaedfe) {
		ldhostobj(ldmacho, f, pkg, len, pn, file);
		return;
	}
	if(c1 == 0x4c && c2 == 0x01 || c1 == 0x64 && c2 == 0x86) {
		ldhostobj(ldpe, f, pkg, len, pn, file);
		return;
	}

	/* check the header */
	line = Brdline(f, '\n');
	if(line == nil) {
		if(Blinelen(f) > 0) {
			diag("%s: not an object file", pn);
			return;
		}
		goto eof;
	}
	n = Blinelen(f) - 1;
	line[n] = '\0';
	if(strncmp(line, "go object ", 10) != 0) {
		if(strlen(pn) > 3 && strcmp(pn+strlen(pn)-3, ".go") == 0) {
			print("%cl: input %s is not .%c file (use %cg to compile .go files)\n", thechar, pn, thechar, thechar);
			errorexit();
		}
		if(strcmp(line, thestring) == 0) {
			// old header format: just $GOOS
			diag("%s: stale object file", pn);
			return;
		}
		diag("%s: not an object file", pn);
		free(pn);
		return;
	}
	
	// First, check that the basic goos, goarch, and version match.
	t = smprint("%s %s %s ", goos, getgoarch(), getgoversion());
	line[n] = ' ';
	if(strncmp(line+10, t, strlen(t)) != 0 && !debug['f']) {
		line[n] = '\0';
		diag("%s: object is [%s] expected [%s]", pn, line+10, t);
		free(t);
		free(pn);
		return;
	}
	
	// Second, check that longer lines match each other exactly,
	// so that the Go compiler and write additional information
	// that must be the same from run to run.
	line[n] = '\0';
	if(n-10 > strlen(t)) {
		if(theline == nil)
			theline = estrdup(line+10);
		else if(strcmp(theline, line+10) != 0) {
			line[n] = '\0';
			diag("%s: object is [%s] expected [%s]", pn, line+10, theline);
			free(t);
			free(pn);
			return;
		}
	}
	free(t);
	line[n] = '\n';

	/* skip over exports and other info -- ends with \n!\n */
	import0 = Boffset(f);
	c1 = '\n';	// the last line ended in \n
	c2 = BGETC(f);
	c3 = BGETC(f);
	while(c1 != '\n' || c2 != '!' || c3 != '\n') {
		c1 = c2;
		c2 = c3;
		c3 = BGETC(f);
		if(c3 == Beof)
			goto eof;
	}
	import1 = Boffset(f);

	Bseek(f, import0, 0);
	ldpkg(f, pkg, import1 - import0 - 2, pn, whence);	// -2 for !\n
	Bseek(f, import1, 0);

	ldobjfile(ctxt, f, pkg, eof - Boffset(f), pn);
	free(pn);
	return;

eof:
	diag("truncated object file: %s", pn);
	free(pn);
}

void
zerosig(char *sp)
{
	LSym *s;

	s = linklookup(ctxt, sp, 0);
	s->sig = 0;
}

void
mywhatsys(void)
{
	goroot = getgoroot();
	goos = getgoos();
	goarch = getgoarch();

	if(strncmp(goarch, thestring, strlen(thestring)) != 0)
		sysfatal("cannot use %cc with GOARCH=%s", thechar, goarch);
}

int
pathchar(void)
{
	return '/';
}

static	uchar*	hunk;
static	uint32	nhunk;
#define	NHUNK	(10UL<<20)

void*
mal(uint32 n)
{
	void *v;

	n = (n+7)&~7;
	if(n > NHUNK) {
		v = malloc(n);
		if(v == nil) {
			diag("out of memory");
			errorexit();
		}
		memset(v, 0, n);
		return v;
	}
	if(n > nhunk) {
		hunk = malloc(NHUNK);
		if(hunk == nil) {
			diag("out of memory");
			errorexit();
		}
		nhunk = NHUNK;
	}

	v = hunk;
	nhunk -= n;
	hunk += n;

	memset(v, 0, n);
	return v;
}

void
unmal(void *v, uint32 n)
{
	n = (n+7)&~7;
	if(hunk - n == v) {
		hunk -= n;
		nhunk += n;
	}
}

// Copied from ../gc/subr.c:/^pathtoprefix; must stay in sync.
/*
 * Convert raw string to the prefix that will be used in the symbol table.
 * Invalid bytes turn into %xx.	 Right now the only bytes that need
 * escaping are %, ., and ", but we escape all control characters too.
 *
 * If you edit this, edit ../gc/subr.c:/^pathtoprefix too.
 * If you edit this, edit ../../debug/goobj/read.go:/importPathToPrefix too.
 */
static char*
pathtoprefix(char *s)
{
	static char hex[] = "0123456789abcdef";
	char *p, *r, *w, *l;
	int n;

	// find first character past the last slash, if any.
	l = s;
	for(r=s; *r; r++)
		if(*r == '/')
			l = r+1;

	// check for chars that need escaping
	n = 0;
	for(r=s; *r; r++)
		if(*r <= ' ' || (*r == '.' && r >= l) || *r == '%' || *r == '"' || *r >= 0x7f)
			n++;

	// quick exit
	if(n == 0)
		return s;

	// escape
	p = mal((r-s)+1+2*n);
	for(r=s, w=p; *r; r++) {
		if(*r <= ' ' || (*r == '.' && r >= l) || *r == '%' || *r == '"' || *r >= 0x7f) {
			*w++ = '%';
			*w++ = hex[(*r>>4)&0xF];
			*w++ = hex[*r&0xF];
		} else
			*w++ = *r;
	}
	*w = '\0';
	return p;
}

int
iconv(Fmt *fp)
{
	char *p;

	p = va_arg(fp->args, char*);
	if(p == nil) {
		fmtstrcpy(fp, "<nil>");
		return 0;
	}
	p = pathtoprefix(p);
	fmtstrcpy(fp, p);
	return 0;
}

Section*
addsection(Segment *seg, char *name, int rwx)
{
	Section **l;
	Section *sect;
	
	for(l=&seg->sect; *l; l=&(*l)->next)
		;
	sect = mal(sizeof *sect);
	sect->rwx = rwx;
	sect->name = name;
	sect->seg = seg;
	sect->align = PtrSize; // everything is at least pointer-aligned
	*l = sect;
	return sect;
}

uint16
le16(uchar *b)
{
	return b[0] | b[1]<<8;
}

uint32
le32(uchar *b)
{
	return b[0] | b[1]<<8 | b[2]<<16 | (uint32)b[3]<<24;
}

uint64
le64(uchar *b)
{
	return le32(b) | (uint64)le32(b+4)<<32;
}

uint16
be16(uchar *b)
{
	return b[0]<<8 | b[1];
}

uint32
be32(uchar *b)
{
	return (uint32)b[0]<<24 | b[1]<<16 | b[2]<<8 | b[3];
}

uint64
be64(uchar *b)
{
	return (uvlong)be32(b)<<32 | be32(b+4);
}

Endian be = { be16, be32, be64 };
Endian le = { le16, le32, le64 };

typedef struct Chain Chain;
struct Chain
{
	LSym *sym;
	Chain *up;
	int limit;  // limit on entry to sym
};

static int stkcheck(Chain*, int);
static void stkprint(Chain*, int);
static void stkbroke(Chain*, int);
static LSym *morestack;
static LSym *newstack;

enum
{
	HasLinkRegister = (thechar == '5' || thechar == '9'),
};

// TODO: Record enough information in new object files to
// allow stack checks here.

static int
callsize(void)
{
	if(HasLinkRegister)
		return 0;
	return RegSize;
}

void
dostkcheck(void)
{
	Chain ch;
	LSym *s;

	morestack = linklookup(ctxt, "runtime.morestack", 0);
	newstack = linklookup(ctxt, "runtime.newstack", 0);

	// Every splitting function ensures that there are at least StackLimit
	// bytes available below SP when the splitting prologue finishes.
	// If the splitting function calls F, then F begins execution with
	// at least StackLimit - callsize() bytes available.
	// Check that every function behaves correctly with this amount
	// of stack, following direct calls in order to piece together chains
	// of non-splitting functions.
	ch.up = nil;
	ch.limit = StackLimit - callsize();

	// Check every function, but do the nosplit functions in a first pass,
	// to make the printed failure chains as short as possible.
	for(s = ctxt->textp; s != nil; s = s->next) {
		// runtime.racesymbolizethunk is called from gcc-compiled C
		// code running on the operating system thread stack.
		// It uses more than the usual amount of stack but that's okay.
		if(strcmp(s->name, "runtime.racesymbolizethunk") == 0)
			continue;

		if(s->nosplit) {
			ctxt->cursym = s;
			ch.sym = s;
			stkcheck(&ch, 0);
		}
	}
	for(s = ctxt->textp; s != nil; s = s->next) {
		if(!s->nosplit) {
			ctxt->cursym = s;
			ch.sym = s;
			stkcheck(&ch, 0);
		}
	}
}

static int
stkcheck(Chain *up, int depth)
{
	Chain ch, ch1;
	LSym *s;
	int limit;
	Reloc *r, *endr;
	Pciter pcsp;
	
	limit = up->limit;
	s = up->sym;
	
	// Don't duplicate work: only need to consider each
	// function at top of safe zone once.
	if(limit == StackLimit-callsize()) {
		if(s->stkcheck)
			return 0;
		s->stkcheck = 1;
	}
	
	if(depth > 100) {
		diag("nosplit stack check too deep");
		stkbroke(up, 0);
		return -1;
	}

	if(s->external || s->pcln == nil) {
		// external function.
		// should never be called directly.
		// only diagnose the direct caller.
		if(depth == 1 && s->type != SXREF)
			diag("call to external function %s", s->name);
		return -1;
	}

	if(limit < 0) {
		stkbroke(up, limit);
		return -1;
	}

	// morestack looks like it calls functions,
	// but it switches the stack pointer first.
	if(s == morestack)
		return 0;

	ch.up = up;
	
	// Walk through sp adjustments in function, consuming relocs.
	r = s->r;
	endr = r + s->nr;
	for(pciterinit(ctxt, &pcsp, &s->pcln->pcsp); !pcsp.done; pciternext(&pcsp)) {
		// pcsp.value is in effect for [pcsp.pc, pcsp.nextpc).

		// Check stack size in effect for this span.
		if(limit - pcsp.value < 0) {
			stkbroke(up, limit - pcsp.value);
			return -1;
		}

		// Process calls in this span.
		for(; r < endr && r->off < pcsp.nextpc; r++) {
			switch(r->type) {
			case R_CALL:
			case R_CALLARM:
			case R_CALLPOWER:
				// Direct call.
				ch.limit = limit - pcsp.value - callsize();
				ch.sym = r->sym;
				if(stkcheck(&ch, depth+1) < 0)
					return -1;

				// If this is a call to morestack, we've just raised our limit back
				// to StackLimit beyond the frame size.
				if(strncmp(r->sym->name, "runtime.morestack", 17) == 0) {
					limit = StackLimit + s->locals;
					if(HasLinkRegister)
						limit += RegSize;
				}
				break;

			case R_CALLIND:
				// Indirect call.  Assume it is a call to a splitting function,
				// so we have to make sure it can call morestack.
				// Arrange the data structures to report both calls, so that
				// if there is an error, stkprint shows all the steps involved.
				ch.limit = limit - pcsp.value - callsize();
				ch.sym = nil;
				ch1.limit = ch.limit - callsize(); // for morestack in called prologue
				ch1.up = &ch;
				ch1.sym = morestack;
				if(stkcheck(&ch1, depth+2) < 0)
					return -1;
				break;
			}
		}
	}
		
	return 0;
}

static void
stkbroke(Chain *ch, int limit)
{
	diag("nosplit stack overflow");
	stkprint(ch, limit);
}

static void
stkprint(Chain *ch, int limit)
{
	char *name;

	if(ch->sym)
		name = ch->sym->name;
	else
		name = "function pointer";

	if(ch->up == nil) {
		// top of chain.  ch->sym != nil.
		if(ch->sym->nosplit)
			print("\t%d\tassumed on entry to %s\n", ch->limit, name);
		else
			print("\t%d\tguaranteed after split check in %s\n", ch->limit, name);
	} else {
		stkprint(ch->up, ch->limit + (!HasLinkRegister)*RegSize);
		if(!HasLinkRegister)
			print("\t%d\ton entry to %s\n", ch->limit, name);
	}
	if(ch->limit != limit)
		print("\t%d\tafter %s uses %d\n", limit, name, ch->limit - limit);
}

int
Yconv(Fmt *fp)
{
	LSym *s;
	Fmt fmt;
	int i;
	char *str;

	s = va_arg(fp->args, LSym*);
	if (s == S) {
		fmtprint(fp, "<nil>");
	} else {
		fmtstrinit(&fmt);
		fmtprint(&fmt, "%s @0x%08llx [%lld]", s->name, (vlong)s->value, (vlong)s->size);
		for (i = 0; i < s->size; i++) {
			if (!(i%8)) fmtprint(&fmt,  "\n\t0x%04x ", i);
			fmtprint(&fmt, "%02x ", s->p[i]);
		}
		fmtprint(&fmt, "\n");
		for (i = 0; i < s->nr; i++) {
			fmtprint(&fmt, "\t0x%04x[%x] %d %s[%llx]\n",
			      s->r[i].off,
			      s->r[i].siz,
			      s->r[i].type,
			      s->r[i].sym->name,
			      (vlong)s->r[i].add);
		}
		str = fmtstrflush(&fmt);
		fmtstrcpy(fp, str);
		free(str);
	}

	return 0;
}

vlong coutpos;

void
cflush(void)
{
	int n;

	if(cbpmax < cbp)
		cbpmax = cbp;
	n = cbpmax - buf.cbuf;
	dowrite(cout, buf.cbuf, n);
	coutpos += n;
	cbp = buf.cbuf;
	cbc = sizeof(buf.cbuf);
	cbpmax = cbp;
}

vlong
cpos(void)
{
	return coutpos + cbp - buf.cbuf;
}

void
cseek(vlong p)
{
	vlong start;
	int delta;

	if(cbpmax < cbp)
		cbpmax = cbp;
	start = coutpos;
	if(start <= p && p <= start+(cbpmax - buf.cbuf)) {
//print("cseek %lld in [%lld,%lld] (%lld)\n", p, start, start+sizeof(buf.cbuf), cpos());
		delta = p - (start + cbp - buf.cbuf);
		cbp += delta;
		cbc -= delta;
//print("now at %lld\n", cpos());
		return;
	}

	cflush();
	seek(cout, p, 0);
	coutpos = p;
}

void
cwrite(void *buf, int n)
{
	cflush();
	if(n <= 0)
		return;
	dowrite(cout, buf, n);
	coutpos += n;
}

void
usage(void)
{
	fprint(2, "usage: %cl [options] main.%c\n", thechar, thechar);
	flagprint(2);
	exits("usage");
}

void
setheadtype(char *s)
{
	int h;
	
	h = headtype(s);
	if(h < 0) {
		fprint(2, "unknown header type -H %s\n", s);
		errorexit();
	}
	headstring = s;
	HEADTYPE = headtype(s);
}

void
setinterp(char *s)
{
	debug['I'] = 1; // denote cmdline interpreter override
	interpreter = s;
}

void
doversion(void)
{
	print("%cl version %s\n", thechar, getgoversion());
	errorexit();
}

void
genasmsym(void (*put)(LSym*, char*, int, vlong, vlong, int, LSym*))
{
	Auto *a;
	LSym *s;
	int32 off;

	// These symbols won't show up in the first loop below because we
	// skip STEXT symbols. Normal STEXT symbols are emitted by walking textp.
	s = linklookup(ctxt, "runtime.text", 0);
	if(s->type == STEXT)
		put(s, s->name, 'T', s->value, s->size, s->version, 0);
	s = linklookup(ctxt, "runtime.etext", 0);
	if(s->type == STEXT)
		put(s, s->name, 'T', s->value, s->size, s->version, 0);

	for(s=ctxt->allsym; s!=S; s=s->allsym) {
		if(s->hide || (s->name[0] == '.' && s->version == 0 && strcmp(s->name, ".rathole") != 0))
			continue;
		switch(s->type&SMASK) {
		case SCONST:
		case SRODATA:
		case SSYMTAB:
		case SPCLNTAB:
		case SDATA:
		case SNOPTRDATA:
		case SELFROSECT:
		case SMACHOGOT:
		case STYPE:
		case SSTRING:
		case SGOSTRING:
		case SWINDOWS:
			if(!s->reachable)
				continue;
			put(s, s->name, 'D', symaddr(s), s->size, s->version, s->gotype);
			continue;

		case SBSS:
		case SNOPTRBSS:
			if(!s->reachable)
				continue;
			if(s->np > 0)
				diag("%s should not be bss (size=%d type=%d special=%d)", s->name, (int)s->np, s->type, s->special);
			put(s, s->name, 'B', symaddr(s), s->size, s->version, s->gotype);
			continue;

		case SFILE:
			put(nil, s->name, 'f', s->value, 0, s->version, 0);
			continue;
		}
	}

	for(s = ctxt->textp; s != nil; s = s->next) {
		put(s, s->name, 'T', s->value, s->size, s->version, s->gotype);

		// NOTE(ality): acid can't produce a stack trace without .frame symbols
		put(nil, ".frame", 'm', s->locals+PtrSize, 0, 0, 0);

		for(a=s->autom; a; a=a->link) {
			// Emit a or p according to actual offset, even if label is wrong.
			// This avoids negative offsets, which cannot be encoded.
			if(a->name != A_AUTO && a->name != A_PARAM)
				continue;
			
			// compute offset relative to FP
			if(a->name == A_PARAM)
				off = a->aoffset;
			else
				off = a->aoffset - PtrSize;
			
			// FP
			if(off >= 0) {
				put(nil, a->asym->name, 'p', off, 0, 0, a->gotype);
				continue;
			}
			
			// SP
			if(off <= -PtrSize) {
				put(nil, a->asym->name, 'a', -(off+PtrSize), 0, 0, a->gotype);
				continue;
			}
			
			// Otherwise, off is addressing the saved program counter.
			// Something underhanded is going on. Say nothing.
		}
	}
	if(debug['v'] || debug['n'])
		Bprint(&bso, "%5.2f symsize = %ud\n", cputime(), symsize);
	Bflush(&bso);
}

vlong
symaddr(LSym *s)
{
	if(!s->reachable)
		diag("unreachable symbol in symaddr - %s", s->name);
	return s->value;
}

void
xdefine(char *p, int t, vlong v)
{
	LSym *s;

	s = linklookup(ctxt, p, 0);
	s->type = t;
	s->value = v;
	s->reachable = 1;
	s->special = 1;
}

vlong
datoff(vlong addr)
{
	if(addr >= segdata.vaddr)
		return addr - segdata.vaddr + segdata.fileoff;
	if(addr >= segtext.vaddr)
		return addr - segtext.vaddr + segtext.fileoff;
	diag("datoff %#llx", addr);
	return 0;
}

vlong
entryvalue(void)
{
	char *a;
	LSym *s;

	a = INITENTRY;
	if(*a >= '0' && *a <= '9')
		return atolwhex(a);
	s = linklookup(ctxt, a, 0);
	if(s->type == 0)
		return INITTEXT;
	if(s->type != STEXT)
		diag("entry not text: %s", s->name);
	return s->value;
}

static void
undefsym(LSym *s)
{
	int i;
	Reloc *r;

	ctxt->cursym = s;
	for(i=0; i<s->nr; i++) {
		r = &s->r[i];
		if(r->sym == nil) // happens for some external ARM relocs
			continue;
		if(r->sym->type == Sxxx || r->sym->type == SXREF)
			diag("undefined: %s", r->sym->name);
		if(!r->sym->reachable)
			diag("use of unreachable symbol: %s", r->sym->name);
	}
}

void
undef(void)
{
	LSym *s;
	
	for(s = ctxt->textp; s != nil; s = s->next)
		undefsym(s);
	for(s = datap; s != nil; s = s->next)
		undefsym(s);
	if(nerrors > 0)
		errorexit();
}

void
callgraph(void)
{
	LSym *s;
	Reloc *r;
	int i;

	if(!debug['c'])
		return;

	for(s = ctxt->textp; s != nil; s = s->next) {
		for(i=0; i<s->nr; i++) {
			r = &s->r[i];
			if(r->sym == nil)
				continue;
			if((r->type == R_CALL || r->type == R_CALLARM || r->type == R_CALLPOWER) && r->sym->type == STEXT)
				Bprint(&bso, "%s calls %s\n", s->name, r->sym->name);
		}
	}
}

void
diag(char *fmt, ...)
{
	char buf[1024], *tn, *sep;
	va_list arg;

	tn = "";
	sep = "";
	if(ctxt->cursym != S) {
		tn = ctxt->cursym->name;
		sep = ": ";
	}
	va_start(arg, fmt);
	vseprint(buf, buf+sizeof(buf), fmt, arg);
	va_end(arg);
	print("%s%s%s\n", tn, sep, buf);

	nerrors++;
	if(nerrors > 20) {
		print("too many errors\n");
		errorexit();
	}
}

void
checkgo(void)
{
	LSym *s;
	Reloc *r;
	int i;
	int changed;
	
	if(!debug['C'])
		return;
	
	// TODO(rsc,khr): Eventually we want to get to no Go-called C functions at all,
	// which would simplify this logic quite a bit.

	// Mark every Go-called C function with cfunc=2, recursively.
	do {
		changed = 0;
		for(s = ctxt->textp; s != nil; s = s->next) {
			if(s->cfunc == 0 || (s->cfunc == 2 && s->nosplit)) {
				for(i=0; i<s->nr; i++) {
					r = &s->r[i];
					if(r->sym == nil)
						continue;
					if((r->type == R_CALL || r->type == R_CALLARM) && r->sym->type == STEXT) {
						if(r->sym->cfunc == 1) {
							changed = 1;
							r->sym->cfunc = 2;
						}
					}
				}
			}
		}
	}while(changed);

	// Complain about Go-called C functions that can split the stack
	// (that can be preempted for garbage collection or trigger a stack copy).
	for(s = ctxt->textp; s != nil; s = s->next) {
		if(s->cfunc == 0 || (s->cfunc == 2 && s->nosplit)) {
			for(i=0; i<s->nr; i++) {
				r = &s->r[i];
				if(r->sym == nil)
					continue;
				if((r->type == R_CALL || r->type == R_CALLARM) && r->sym->type == STEXT) {
					if(s->cfunc == 0 && r->sym->cfunc == 2 && !r->sym->nosplit)
						print("Go %s calls C %s\n", s->name, r->sym->name);
					else if(s->cfunc == 2 && s->nosplit && !r->sym->nosplit)
						print("Go calls C %s calls %s\n", s->name, r->sym->name);
				}
			}
		}
	}
}
