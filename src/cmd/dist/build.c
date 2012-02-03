// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "a.h"

/*
 * Initialization for any invocation.
 */

// The usual variables.
char *goarch;
char *gobin;
char *gohostarch;
char *gohostos;
char *goos;
char *goroot;
char *workdir;
char *gochar;
char *goroot_final;
char *goversion = "go1";  // TODO: Read correct version
char *slash;	// / for unix, \ for windows
char *default_goroot;

static void fixslash(Buf*);
static bool shouldbuild(char*, char*);
static void copy(char*, char*);

// The known architecture letters.
static char *gochars = "568";

// The known architectures.
static char *okgoarch[] = {
	// same order as gochars
	"arm",
	"amd64",
	"386",
};

// The known operating systems.
static char *okgoos[] = {
	"darwin",
	"linux",
	"freebsd",
	"netbsd",
	"openbsd",
	"plan9",
	"windows",
};

static void rmworkdir(void);

// find reports the first index of p in l[0:n], or else -1.
static int
find(char *p, char **l, int n)
{
	int i;
	
	for(i=0; i<n; i++)
		if(streq(p, l[i]))
			return i;
	return -1;
}

// init handles initialization of the various global state, like goroot and goarch.
void
init(void)
{
	char *p;
	int i;
	Buf b;
	
	binit(&b);

	xgetenv(&b, "GOROOT");
	if(b.len == 0) {
		if(default_goroot == nil)
			fatal("$GOROOT not set and not available");
		bwritestr(&b, default_goroot);
	}
	goroot = btake(&b);

	xgetenv(&b, "GOBIN");
	if(b.len == 0)
		bprintf(&b, "%s%sbin", goroot, slash);
	gobin = btake(&b);

	xgetenv(&b, "GOOS");
	if(b.len == 0)
		bwritestr(&b, gohostos);
	goos = btake(&b);
	if(find(goos, okgoos, nelem(okgoos)) < 0)
		fatal("unknown $GOOS %s", goos);

	p = bprintf(&b, "%s/include/u.h", goroot);
	fixslash(&b);
	if(!isfile(p)) {
		fatal("$GOROOT is not set correctly or not exported\n"
			"\tGOROOT=%s\n"
			"\t%s does not exist", goroot, p);
	}
	
	xgetenv(&b, "GOHOSTARCH");
	if(b.len > 0)
		gohostarch = btake(&b);

	if(find(gohostarch, okgoarch, nelem(okgoarch)) < 0)
		fatal("unknown $GOHOSTARCH %s", gohostarch);

	xgetenv(&b, "GOARCH");
	if(b.len == 0)
		bwritestr(&b, gohostarch);
	goarch = btake(&b);
	if((i=find(goarch, okgoarch, nelem(okgoarch))) < 0)
		fatal("unknown $GOARCH %s", goarch);
	bprintf(&b, "%c", gochars[i]);
	gochar = btake(&b);

	xgetenv(&b, "GOROOT_FINAL");
	if(b.len > 0)
		goroot_final = btake(&b);
	else
		goroot_final = goroot;
	
	xsetenv("GOROOT", goroot);
	xsetenv("GOARCH", goarch);
	xsetenv("GOOS", goos);
	
	// Make the environment more predictable.
	xsetenv("LANG", "C");
	xsetenv("LANGUAGE", "en_US.UTF8");

	workdir = xworkdir();
	xatexit(rmworkdir);

	bfree(&b);
}

// rmworkdir deletes the work directory.
static void
rmworkdir(void)
{
	xprintf("rm -rf %s\n", workdir);
	xremoveall(workdir);
}

/*
 * Initial tree setup.
 */

// The old tools that no longer live in $GOBIN or $GOROOT/bin.
static char *oldtool[] = {
	"5a", "5c", "5g", "5l",
	"6a", "6c", "6g", "6l",
	"8a", "8c", "8g", "8l",
	"6cov",
	"6nm",
	"cgo",
	"ebnflint",
	"goapi",
	"gofix",
	"goinstall",
	"gomake",
	"gopack",
	"gopprof",
	"gotest",
	"gotype",
	"govet",
	"goyacc",
	"quietgcc",
};

// setup sets up the tree for the initial build.
static void
setup(void)
{
	int i;
	Buf b;
	char *p;

	binit(&b);

	run(&b, nil, 0, "ld", "--version", nil);
	if(contains(bstr(&b), "gold") && contains(bstr(&b), " 2.20")) {
		fatal("Your system has gold 2.20 installed.\n"
			"This version is shipped by Ubuntu even though\n"
			"it is known not to work on Ubuntu.\n"
			"Binaries built with this linker are likely to fail in mysterious ways.\n"
			"\n"
			"Run sudo apt-get remove binutils-gold.");
	}

	// Create tool directory.
	p = bprintf(&b, "%s/bin", goroot);
	fixslash(&b);
	if(!isdir(p))
		xmkdir(p);
	p = bprintf(&b, "%s/bin/go-tool", goroot);
	fixslash(&b);
	if(!isdir(p))
		xmkdir(p);

	// Create package directory.
	p = bprintf(&b, "%s/pkg", goroot);
	fixslash(&b);
	if(!isdir(p))
		xmkdir(p);
	p = bprintf(&b, "%s/pkg/%s_%s", goroot, goos, goarch);
	fixslash(&b);
	xremoveall(p);
	xmkdir(p);

	// Remove old pre-tool binaries.
	for(i=0; i<nelem(oldtool); i++)
		xremove(bprintf(&b, "%s%s%s%s%s", goroot, slash, "bin", slash, oldtool[i]));
	
	// If $GOBIN is set and has a Go compiler, it must be cleaned.
	for(i=0; gochars[i]; i++) {
		if(isfile(bprintf(&b, "%s%s%c%s", gobin, slash, gochars[i], "g"))) {
			for(i=0; i<nelem(oldtool); i++)
				xremove(bprintf(&b, "%s%s%s", gobin, slash, oldtool[i]));
			break;
		}
	}

	bfree(&b);
}

/*
 * C library and tool building
 */

// gccargs is the gcc command line to use for compiling a single C file.
static char *gccargs[] = {
	"gcc",
	"-Wall",
	"-Wno-sign-compare",
	"-Wno-missing-braces",
	"-Wno-parentheses",
	"-Wno-unknown-pragmas",
	"-Wno-switch",
	"-Wno-comment",
	"-Werror",
	"-fno-common",
	"-ggdb",
	"-O2",
	"-c",
};

// deptab lists changes to the default dependencies for a given prefix.
// deps ending in /* read the whole directory; deps beginning with - 
// exclude files with that prefix.
static struct {
	char *prefix;  // prefix of target
	char *dep[20];  // dependency tweaks for targets with that prefix
} deptab[] = {
	{"lib9", {
		"$GOROOT/include/u.h",
		"$GOROOT/include/utf.h",
		"$GOROOT/include/fmt.h",
		"$GOROOT/include/libc.h",
		"fmt/*",
		"utf/*",
		"-utf/mkrunetype",
		"-utf\\mkrunetype",
		"-utf/runetypebody",
		"-utf\\runetypebody",
	}},
	{"libbio", {
		"$GOROOT/include/u.h",
		"$GOROOT/include/utf.h",
		"$GOROOT/include/fmt.h",
		"$GOROOT/include/libc.h",
		"$GOROOT/include/bio.h",
	}},
	{"libmach", {
		"$GOROOT/include/u.h",
		"$GOROOT/include/utf.h",
		"$GOROOT/include/fmt.h",
		"$GOROOT/include/libc.h",
		"$GOROOT/include/bio.h",
		"$GOROOT/include/ar.h",
		"$GOROOT/include/bootexec.h",
		"$GOROOT/include/mach.h",
		"$GOROOT/include/ureg_amd64.h",
		"$GOROOT/include/ureg_arm.h",
		"$GOROOT/include/ureg_x86.h",
	}},
	{"cmd/cc", {
		"-pgen.c",
		"-pswt.c",
	}},
	{"cmd/gc", {
		"-cplx.c",
		"-pgen.c",
		"-y1.tab.c",  // makefile dreg
		"opnames.h",
	}},
	{"cmd/5c", {
		"../cc/pgen.c",
		"../cc/pswt.c",
		"../5l/enam.c",
		"$GOROOT/lib/libcc.a",
	}},
	{"cmd/6c", {
		"../cc/pgen.c",
		"../cc/pswt.c",
		"../6l/enam.c",
		"$GOROOT/lib/libcc.a",
	}},
	{"cmd/8c", {
		"../cc/pgen.c",
		"../cc/pswt.c",
		"../8l/enam.c",
		"$GOROOT/lib/libcc.a",
	}},
	{"cmd/5g", {
		"../gc/cplx.c",
		"../gc/pgen.c",
		"../5l/enam.c",
		"$GOROOT/lib/libgc.a",
	}},
	{"cmd/6g", {
		"../gc/cplx.c",
		"../gc/pgen.c",
		"../6l/enam.c",
		"$GOROOT/lib/libgc.a",
	}},
	{"cmd/8g", {
		"../gc/cplx.c",
		"../gc/pgen.c",
		"../8l/enam.c",
		"$GOROOT/lib/libgc.a",
	}},
	{"cmd/5l", {
		"../ld/*",
		"enam.c",
	}},
	{"cmd/6l", {
		"../ld/*",
		"enam.c",
	}},
	{"cmd/8l", {
		"../ld/*",
		"enam.c",
	}},
	{"cmd/", {
		"$GOROOT/lib/libmach.a",
		"$GOROOT/lib/libbio.a",
		"$GOROOT/lib/lib9.a",
	}},
};

// depsuffix records the allowed suffixes for source files.
char *depsuffix[] = {
	".c",
	".h",
	".s",
	".go",
};

// gentab records how to generate some trivial files.
static struct {
	char *name;
	void (*gen)(char*, char*);
} gentab[] = {
	{"opnames.h", gcopnames},
	{"enam.c", mkenam},
};

// install installs the library, package, or binary associated with dir,
// which is relative to $GOROOT/src.
static void
install(char *dir)
{
	char *name, *p, *elem, *prefix;
	bool islib, ispkg, isgo, stale;
	Buf b, b1, path;
	Vec compile, files, link, go, missing, clean, lib, extra;
	Time ttarg, t;
	int i, j, k, n;

	binit(&b);
	binit(&b1);
	binit(&path);
	vinit(&compile);
	vinit(&files);
	vinit(&link);
	vinit(&go);
	vinit(&missing);
	vinit(&clean);
	vinit(&lib);
	vinit(&extra);
	
	// path = full path to dir.
	bprintf(&path, "%s/src/%s", goroot, dir);
	fixslash(&path);
	name = lastelem(dir);

	islib = hasprefix(dir, "lib") || streq(dir, "cmd/cc") || streq(dir, "cmd/gc");
	ispkg = hasprefix(dir, "pkg");
	isgo = ispkg || streq(dir, "cmd/go");

	
	// Start final link command line.
	// Note: code below knows that link.p[2] is the target.
	if(islib) {
		// C library.
		vadd(&link, "ar");
		vadd(&link, "rsc");
		prefix = "";
		if(!hasprefix(name, "lib"))
			prefix = "lib";
		bprintf(&b, "%s/lib/%s%s.a", goroot, prefix, name);
		fixslash(&b);
		vadd(&link, bstr(&b));
	} else if(ispkg) {
		// Go library (package).
		bprintf(&b, "%s/bin/go-tool/pack", goroot);
		fixslash(&b);
		vadd(&link, bstr(&b));
		vadd(&link, "grc");
		p = bprintf(&b, "%s/pkg/%s_%s/%s", goroot, goos, goarch, dir+4);
		*xstrrchr(p, '/') = '\0';
		xmkdirall(p);
		bprintf(&b, "%s/pkg/%s_%s/%s.a", goroot, goos, goarch, dir+4);
		fixslash(&b);
		vadd(&link, bstr(&b));
	} else if(streq(dir, "cmd/go")) {
		// Go command.
		bprintf(&b, "%s/bin/go-tool/%sl", goroot, gochar);
		fixslash(&b);
		vadd(&link, bstr(&b));
		vadd(&link, "-o");
		bprintf(&b, "%s/bin/go-tool/go_bootstrap", goroot);
		fixslash(&b);
		vadd(&link, bstr(&b));
	} else {
		// C command.
		vadd(&link, "gcc");
		vadd(&link, "-o");
		bprintf(&b, "%s/bin/go-tool/%s", goroot, name);
		fixslash(&b);
		vadd(&link, bstr(&b));
	}
	ttarg = mtime(link.p[2]);

	// Gather files that are sources for this target.
	// Everything in that directory, and any target-specific
	// additions.
	xreaddir(&files, bstr(&path));
	for(i=0; i<nelem(deptab); i++) {
		if(hasprefix(dir, deptab[i].prefix)) {
			for(j=0; (p=deptab[i].dep[j])!=nil; j++) {
				if(hasprefix(p, "$GOROOT/")) {
					bprintf(&b1, "%s/%s", goroot, p+8);
					p = bstr(&b1);
				}
				if(hassuffix(p, ".a")) {
					vadd(&lib, p);
					continue;
				}
				if(hassuffix(p, "/*")) {
					bprintf(&b, "%s/%s", bstr(&path), p);
					b.len -= 2;
					fixslash(&b);
					xreaddir(&extra, bstr(&b));
					bprintf(&b, "%s", p);
					b.len -= 2;
					for(k=0; k<extra.len; k++) {
						bprintf(&b1, "%s/%s", bstr(&b), extra.p[k]);
						fixslash(&b1);
						vadd(&files, bstr(&b1));
					}
					continue;
				}
				if(hasprefix(p, "-")) {
					p++;
					n = 0;
					for(k=0; k<files.len; k++) {
						if(hasprefix(files.p[k], p))
							xfree(files.p[k]);
						else
							files.p[n++] = files.p[k];
					}
					files.len = n;
					continue;
				}				
				vadd(&files, p);
			}
		}
	}
	vuniq(&files);

	// Convert to absolute paths.
	for(i=0; i<files.len; i++) {
		if(!isabs(files.p[i])) {
			bprintf(&b, "%s/%s", bstr(&path), files.p[i]);
			fixslash(&b);
			xfree(files.p[i]);
			files.p[i] = btake(&b);
		}
	}

	// For package runtime, copy some files into the work space.
	if(streq(dir, "pkg/runtime")) {		
		copy(bprintf(&b, "%s/arch_GOARCH.h", workdir),
			bprintf(&b1, "%s/arch_%s.h", bstr(&path), goarch));
		copy(bprintf(&b, "%s/defs_GOOS_GOARCH.h", workdir),
			bprintf(&b1, "%s/defs_%s_%s.h", bstr(&path), goos, goarch));
		copy(bprintf(&b, "%s/os_GOOS.h", workdir),
			bprintf(&b1, "%s/os_%s.h", bstr(&path), goos));
		copy(bprintf(&b, "%s/signals_GOOS.h", workdir),
			bprintf(&b1, "%s/signals_%s.h", bstr(&path), goos));
		copy(bprintf(&b, "%s/zasm_GOOS_GOARCH.h", workdir),
			bprintf(&b1, "%s/zasm_%s_%s.h", bstr(&path), goos, goarch));
	}


	// Is the target up-to-date?
	stale = 1;  // TODO: Decide when 0 is okay.
	n = 0;
	for(i=0; i<files.len; i++) {
		p = files.p[i];
		for(j=0; j<nelem(depsuffix); j++)
			if(hassuffix(p, depsuffix[j]))
				goto ok;
		xfree(files.p[i]);
		continue;
	ok:
		t = mtime(p);
		if(t > ttarg)
			stale = 1;
		if(t == 0) {
			vadd(&missing, p);
			files.p[n++] = files.p[i];
			continue;
		}
		if(!hassuffix(p, ".a") && !shouldbuild(p, dir)) {
			xfree(files.p[i]);
			continue;
		}
		if(hassuffix(p, ".go"))
			vadd(&go, p);
		files.p[n++] = files.p[i];
	}
	files.len = n;
	
	for(i=0; i<lib.len && !stale; i++)
		if(mtime(lib.p[i]) > ttarg)
			stale = 1;
		
	if(!stale)
		goto out;

	// Generate any missing files.
	for(i=0; i<missing.len; i++) {
		p = missing.p[i];
		elem = lastelem(p);
		for(j=0; j<nelem(gentab); j++) {
			if(streq(gentab[j].name, elem)) {
				gentab[j].gen(bstr(&path), p);
				vadd(&clean, p);
				goto built;
			}
		}
		fatal("missing file %s", p);
	built:;
	}

	// Compile the files.
	for(i=0; i<files.len; i++) {
		if(!hassuffix(files.p[i], ".c") && !hassuffix(files.p[i], ".s"))
			continue;
		name = lastelem(files.p[i]);

		vreset(&compile);
		if(!isgo) {
			// C library or tool.
			vcopy(&compile, gccargs, nelem(gccargs));
			if(streq(gohostarch, "amd64"))
				vadd(&compile, "-m64");
			else if(streq(gohostarch, "386"))
				vadd(&compile, "-m32");
			if(streq(dir, "lib9"))
				vadd(&compile, "-DPLAN9PORT");
	
			bprintf(&b, "%s/include", goroot);
			fixslash(&b);
			vadd(&compile, "-I");
			vadd(&compile, bstr(&b));
			
			vadd(&compile, "-I");
			vadd(&compile, bstr(&path));
	
			// runtime/goos.c gets the default constants hard-coded.
			if(streq(name, "goos.c")) {
				vadd(&compile, bprintf(&b, "-DGOOS=\"%s\"", goos));
				vadd(&compile, bprintf(&b, "-DGOARCH=\"%s\"", goarch));
				vadd(&compile, bprintf(&b, "-DGOROOT=\"%s\"", goroot));
				vadd(&compile, bprintf(&b, "-DGOVERSION=\"%s\"", goversion));
			}
	
			// gc/lex.c records the GOEXPERIMENT setting used during the build.
			if(streq(name, "lex.c")) {
				xgetenv(&b, "GOEXPERIMENT");
				vadd(&compile, bprintf(&b1, "-DGOEXPERIMENT=\"%s\"", bstr(&b)));
			}
		} else {
			// Supporting files for a Go package.
			if(hassuffix(files.p[i], ".s")) {
				bprintf(&b, "%s/bin/go-tool/%sa", goroot, gochar);
				fixslash(&b);
				vadd(&compile, bstr(&b));
			} else {
				bprintf(&b, "%s/bin/go-tool/%sc", goroot, gochar);
				fixslash(&b);
				vadd(&compile, bstr(&b));
				vadd(&compile, "-FVw");
			}
			vadd(&compile, "-I");
			vadd(&compile, workdir);
			vadd(&compile, bprintf(&b, "-DGOOS_%s", goos));
			vadd(&compile, bprintf(&b, "-DGOARCH_%s", goos));
		}	

		bprintf(&b, "%s/%s", workdir, lastelem(files.p[i]));
		b.p[b.len-1] = 'o';  // was c or s
		fixslash(&b);
		vadd(&compile, "-o");
		vadd(&compile, bstr(&b));
		vadd(&link, bstr(&b));
		vadd(&clean, bstr(&b));

		vadd(&compile, files.p[i]);

		runv(nil, bstr(&path), CheckExit, &compile);
		vreset(&compile);
	}
	
	if(isgo) {
		// The last loop was compiling individual files.
		// Hand the Go files to the compiler en masse.
		vreset(&compile);
		bprintf(&b, "%s/bin/go-tool/%sg", goroot, gochar);
		fixslash(&b);
		vadd(&compile, bstr(&b));

		bprintf(&b, "%s/_go_.%s", workdir, gochar);
		fixslash(&b);
		vadd(&compile, "-o");
		vadd(&compile, bstr(&b));
		vadd(&clean, bstr(&b));
		vadd(&link, bstr(&b));
		
		vadd(&compile, "-p");
		if(hasprefix(dir, "pkg/"))
			vadd(&compile, dir+4);
		else
			vadd(&compile, "main");
		
		if(streq(dir, "pkg/runtime"))
			vadd(&compile, "-+");
		
		vcopy(&compile, go.p, go.len);

		runv(nil, bstr(&path), CheckExit, &compile);
	}

	if(!islib && !isgo) {
		// C binaries need the libraries explicitly, and -lm.
		vcopy(&link, lib.p, lib.len);
		vadd(&link, "-lm");
	}

	// Remove target before writing it.
	xremove(link.p[2]);

	runv(nil, nil, CheckExit, &link);

out:
	for(i=0; i<clean.len; i++)
		xremove(clean.p[i]);

	bfree(&b);
	bfree(&b1);
	bfree(&path);
	vfree(&compile);
	vfree(&files);
	vfree(&link);
	vfree(&go);
	vfree(&missing);
	vfree(&clean);
	vfree(&lib);
	vfree(&extra);
}

// matchfield reports whether the field matches this build.
static bool
matchfield(char *f)
{
	return streq(f, goos) || streq(f, goarch) || streq(f, "cmd_go_bootstrap");
}

// shouldbuild reports whether we should build this file.
// It applies the same rules that are used with context tags
// in package go/build, except that the GOOS and GOARCH
// can appear anywhere in the file name, not just after _.
// In particular, they can be the entire file name (like windows.c).
// We also allow the special tag cmd_go_bootstrap.
// See ../go/bootstrap.go and package go/build.
static bool
shouldbuild(char *file, char *dir)
{
	char *name, *p;
	int i, j, ret, true;
	Buf b;
	Vec lines, fields;
	
	// Check file name for GOOS or GOARCH.
	name = lastelem(file);
	for(i=0; i<nelem(okgoos); i++)
		if(contains(name, okgoos[i]) && !streq(okgoos[i], goos))
			return 0;
	for(i=0; i<nelem(okgoarch); i++)
		if(contains(name, okgoarch[i]) && !streq(okgoarch[i], goarch))
			return 0;
	
	// Omit test files.
	if(contains(name, "_test"))
		return 0;

	// Check file contents for // +build lines.
	binit(&b);
	vinit(&lines);
	vinit(&fields);

	ret = 1;
	readfile(&b, file);
	splitlines(&lines, bstr(&b));
	for(i=0; i<lines.len; i++) {
		p = lines.p[i];
		while(*p == ' ' || *p == '\t' || *p == '\r' || *p == '\n')
			p++;
		if(*p == '\0')
			continue;
		if(contains(p, "package documentation")) {
			ret = 0;
			goto out;
		}
		if(contains(p, "package main") && !streq(dir, "cmd/go")) {
			ret = 0;
			goto out;
		}
		if(!hasprefix(p, "//"))
			break;
		if(!contains(p, "+build"))
			continue;
		splitfields(&fields, lines.p[i]);
		if(fields.len < 2 || !streq(fields.p[1], "+build"))
			continue;
		for(j=2; j<fields.len; j++) {
			p = fields.p[j];
			if((*p == '!' && !matchfield(p+1)) || matchfield(p))
				goto fieldmatch;
		}
		ret = 0;
		goto out;
	fieldmatch:;
	}

out:
	bfree(&b);
	vfree(&lines);
	vfree(&fields);
	
	return ret;
}

// fixslash rewrites / to \ when the slash character is \, so that the paths look conventional.
static void
fixslash(Buf *b)
{
	int i;

	if(slash[0] == '/')
		return;
	for(i=0; i<b->len; i++)
		if(b->p[i] == '/')
			b->p[i] = '\\';
}

// copy copies the file src to dst, via memory (so only good for small files).
static void
copy(char *dst, char *src)
{
	Buf b;
	
	binit(&b);
	readfile(&b, src);
	writefile(&b, dst);
	bfree(&b);
}

/*
 * command implementations
 */

// The env command prints the default environment.
void
cmdenv(int argc, char **argv)
{
	USED(argc);
	USED(argv);
	
	xprintf("GOROOT=%s\n", goroot);
	xprintf("GOARCH=%s\n", goarch);
	xprintf("GOOS=%s\n", goos);
}

// buildorder records the order of builds for the 'go bootstrap' command.
static char *buildorder[] = {
	"lib9",
	"libbio",
	"libmach",

	"cmd/cov",
	"cmd/nm",
	"cmd/pack",
	"cmd/prof",

	"cmd/cc",  // must be before c
	"cmd/gc",  // must be before g
	"cmd/%sl",  // must be before a, c, g
	"cmd/%sa",
	"cmd/%sc",
	"cmd/%sg",

	// The dependency order here was copied from a buildscript
	// back when there were build scripts.  Will have to
	// be maintained by hand, but shouldn't change very
	// often.
	"pkg/runtime",
	"pkg/errors",
	"pkg/sync/atomic",
	"pkg/sync",
	"pkg/io",
	"pkg/unicode",
	"pkg/unicode/utf8",
	"pkg/unicode/utf16",
	"pkg/bytes",
	"pkg/math",
	"pkg/strings",
	"pkg/strconv",
	"pkg/bufio",
	"pkg/sort",
	"pkg/container/heap",
	"pkg/encoding/base64",
	"pkg/syscall",
	"pkg/time",
	"pkg/os",
	"pkg/reflect",
	"pkg/fmt",
	"pkg/encoding/json",
	"pkg/encoding/gob",
	"pkg/flag",
	"pkg/path/filepath",
	"pkg/path",
	"pkg/io/ioutil",
	"pkg/log",
	"pkg/regexp/syntax",
	"pkg/regexp",
	"pkg/go/token",
	"pkg/go/scanner",
	"pkg/go/ast",
	"pkg/go/parser",
	"pkg/go/build",
	"pkg/os/exec",
	"pkg/net/url",
	"pkg/text/template/parse",
	"pkg/text/template",

	"cmd/go",
};

// The bootstrap command runs a build from scratch,
// stopping at having installed the go_bootstrap command.
void
cmdbootstrap(int argc, char **argv)
{
	int i;
	Buf b;
	char *p;

	setup();
	
	// TODO: nuke();
	
	binit(&b);
	for(i=0; i<nelem(buildorder); i++) {
		p = bprintf(&b, buildorder[i], gochar);
		xprintf("%s\n", p);
		install(p);
	}
	bfree(&b);
}

// Install installs the list of packages named on the command line.
void
cmdinstall(int argc, char **argv)
{
	int i;

	for(i=1; i<argc; i++)
		install(argv[i]);
}
