// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "a.h"
#include "arg.h"

/*
 * Initialization for any invocation.
 */

// The usual variables.
char *goarch;
char *gobin;
char *gohostarch;
char *gohostchar;
char *gohostos;
char *goos;
char *goarm;
char *go386;
char *goroot = GOROOT_FINAL;
char *goroot_final = GOROOT_FINAL;
char *goextlinkenabled = "";
char *workdir;
char *tooldir;
char *gochar;
char *goversion;
char *slash;	// / for unix, \ for windows
char *defaultcc;
char *defaultcflags;
char *defaultldflags;
char *defaultcxxtarget;
char *defaultcctarget;
bool	rebuildall;
bool defaultclang;

static bool shouldbuild(char*, char*);
static void dopack(char*, char*, char**, int);
static char *findgoversion(void);

// The known architecture letters.
static char *gochars = "5668";

// The known architectures.
static char *okgoarch[] = {
	// same order as gochars
	"arm",
	"amd64",
	"amd64p32",
	"386",
};

// The known operating systems.
static char *okgoos[] = {
	"darwin",
	"dragonfly",
	"linux",
	"android",
	"solaris",
	"freebsd",
	"nacl",
	"netbsd",
	"openbsd",
	"plan9",
	"windows",
};

static void rmworkdir(void);

// find reports the first index of p in l[0:n], or else -1.
int
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
	if(b.len > 0) {
		// if not "/", then strip trailing path separator
		if(b.len >= 2 && b.p[b.len - 1] == slash[0])
			b.len--;
		goroot = btake(&b);
	}

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

	xgetenv(&b, "GOARM");
	if(b.len == 0)
		bwritestr(&b, xgetgoarm());
	goarm = btake(&b);

	xgetenv(&b, "GO386");
	if(b.len == 0) {
		if(cansse2())
			bwritestr(&b, "sse2");
		else
			bwritestr(&b, "387");
	}
	go386 = btake(&b);

	p = bpathf(&b, "%s/include/u.h", goroot);
	if(!isfile(p)) {
		fatal("$GOROOT is not set correctly or not exported\n"
			"\tGOROOT=%s\n"
			"\t%s does not exist", goroot, p);
	}

	xgetenv(&b, "GOHOSTARCH");
	if(b.len > 0)
		gohostarch = btake(&b);

	i = find(gohostarch, okgoarch, nelem(okgoarch));
	if(i < 0)
		fatal("unknown $GOHOSTARCH %s", gohostarch);
	bprintf(&b, "%c", gochars[i]);
	gohostchar = btake(&b);

	xgetenv(&b, "GOARCH");
	if(b.len == 0)
		bwritestr(&b, gohostarch);
	goarch = btake(&b);
	i = find(goarch, okgoarch, nelem(okgoarch));
	if(i < 0)
		fatal("unknown $GOARCH %s", goarch);
	bprintf(&b, "%c", gochars[i]);
	gochar = btake(&b);

	xgetenv(&b, "GO_EXTLINK_ENABLED");
	if(b.len > 0) {
		goextlinkenabled = btake(&b);
		if(!streq(goextlinkenabled, "0") && !streq(goextlinkenabled, "1"))
			fatal("unknown $GO_EXTLINK_ENABLED %s", goextlinkenabled);
	}
	
	xgetenv(&b, "CC");
	if(b.len == 0) {
		// Use clang on OS X, because gcc is deprecated there.
		// Xcode for OS X 10.9 Mavericks will ship a fake "gcc" binary that
		// actually runs clang. We prepare different command
		// lines for the two binaries, so it matters what we call it.
		// See golang.org/issue/5822.
		if(defaultclang)
			bprintf(&b, "clang");
		else
			bprintf(&b, "gcc");
	}
	defaultcc = btake(&b);

	xgetenv(&b, "CFLAGS");
	defaultcflags = btake(&b);

	xgetenv(&b, "LDFLAGS");
	defaultldflags = btake(&b);

	xgetenv(&b, "CC_FOR_TARGET");
	if(b.len == 0) {
		bprintf(&b, defaultcc);
	}
	defaultcctarget = btake(&b);

	xgetenv(&b, "CXX_FOR_TARGET");
	if(b.len == 0) {
		xgetenv(&b, "CXX");
		if(b.len == 0) {
			if(defaultclang)
				bprintf(&b, "clang++");
			else
				bprintf(&b, "g++");
		}
	}
	defaultcxxtarget = btake(&b);

	xsetenv("GOROOT", goroot);
	xsetenv("GOARCH", goarch);
	xsetenv("GOOS", goos);
	xsetenv("GOARM", goarm);
	xsetenv("GO386", go386);

	// Make the environment more predictable.
	xsetenv("LANG", "C");
	xsetenv("LANGUAGE", "en_US.UTF8");

	goversion = findgoversion();

	workdir = xworkdir();
	xatexit(rmworkdir);

	bpathf(&b, "%s/pkg/tool/%s_%s", goroot, gohostos, gohostarch);
	tooldir = btake(&b);

	bfree(&b);
}

// rmworkdir deletes the work directory.
static void
rmworkdir(void)
{
	if(vflag > 1)
		errprintf("rm -rf %s\n", workdir);
	xremoveall(workdir);
}

// Remove trailing spaces.
static void
chomp(Buf *b)
{
	int c;

	while(b->len > 0 && ((c=b->p[b->len-1]) == ' ' || c == '\t' || c == '\r' || c == '\n'))
		b->len--;
}

static char*
branchtag(char *branch, bool *precise)
{
	char *tag, *p, *q;
	int i;
	Buf b, arg;
	Vec tags;

	binit(&b);
	binit(&arg);
	vinit(&tags);

	bprintf(&arg, "master..%s", branch);
	run(&b, goroot, CheckExit, "git", "log", "--decorate=full", "--format=format:%d", bstr(&arg), nil);

	splitlines(&tags, bstr(&b));
	tag = branch;
	for(i=0; i < tags.len; i++) {
		// Each line is either blank, or looks like
		//	  (tag: refs/tags/go1.4rc2, refs/remotes/origin/release-branch.go1.4, refs/heads/release-branch.go1.4)
		// We need to find an element starting with refs/tags/.
		p = xstrstr(tags.p[i], " refs/tags/");
		if(p == nil)
			continue;
		p += xstrlen(" refs/tags/");
		// The tag name ends at a comma or paren (prefer the first).
		q = xstrstr(p, ",");
		if(q == nil)
			q = xstrstr(p, ")");
		if(q == nil)
			continue;  // malformed line; ignore it
		*q = '\0';
		tag = xstrdup(p);
		if(i == 0)
			*precise = 1;  // tag denotes HEAD
		break;
	}

	bfree(&b);
	bfree(&arg);
	vfree(&tags);
	return tag;
}

// findgoversion determines the Go version to use in the version string.
static char*
findgoversion(void)
{
	char *tag, *p;
	bool precise;
	Buf b, path, bmore, branch;

	binit(&b);
	binit(&path);
	binit(&bmore);
	binit(&branch);

	// The $GOROOT/VERSION file takes priority, for distributions
	// without the source repo.
	bpathf(&path, "%s/VERSION", goroot);
	if(isfile(bstr(&path))) {
		readfile(&b, bstr(&path));
		chomp(&b);
		// Commands such as "dist version > VERSION" will cause
		// the shell to create an empty VERSION file and set dist's
		// stdout to its fd. dist in turn looks at VERSION and uses
		// its content if available, which is empty at this point.
		if(b.len > 0)
			goto done;
	}

	// The $GOROOT/VERSION.cache file is a cache to avoid invoking
	// git every time we run this command.  Unlike VERSION, it gets
	// deleted by the clean command.
	bpathf(&path, "%s/VERSION.cache", goroot);
	if(isfile(bstr(&path))) {
		readfile(&b, bstr(&path));
		chomp(&b);
		goto done;
	}

	// Otherwise, use Git.
	// What is the current branch?
	run(&branch, goroot, CheckExit, "git", "rev-parse", "--abbrev-ref", "HEAD", nil);
	chomp(&branch);

	// What are the tags along the current branch?
	tag = "devel";
	precise = 0;

	// If we're on a release branch, use the closest matching tag
	// that is on the release branch (and not on the master branch).
	if(hasprefix(bstr(&branch), "release-branch."))
		tag = branchtag(bstr(&branch), &precise);

	bprintf(&b, "%s", tag);
	if(!precise) {
		// Tag does not point at HEAD; add hash and date to version.
		run(&bmore, goroot, CheckExit, "git", "log", "-n", "1", "--format=format: +%h %cd", "HEAD", nil);
		chomp(&bmore);
		bwriteb(&b, &bmore);
	}

	// Cache version.
	writefile(&b, bstr(&path), 0);

done:
	p = btake(&b);


	bfree(&b);
	bfree(&path);
	bfree(&bmore);
	bfree(&branch);

	return p;
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
	"6prof",
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

// Unreleased directories (relative to $GOROOT) that should
// not be in release branches.
static char *unreleased[] = {
	"src/cmd/link",
	"src/debug/goobj",
	"src/old",
};

// setup sets up the tree for the initial build.
static void
setup(void)
{
	int i;
	Buf b;
	char *p;

	binit(&b);

	// Create bin directory.
	p = bpathf(&b, "%s/bin", goroot);
	if(!isdir(p))
		xmkdir(p);

	// Create package directory.
	p = bpathf(&b, "%s/pkg", goroot);
	if(!isdir(p))
		xmkdir(p);
	p = bpathf(&b, "%s/pkg/%s_%s", goroot, gohostos, gohostarch);
	if(rebuildall)
		xremoveall(p);
	xmkdirall(p);
	if(!streq(goos, gohostos) || !streq(goarch, gohostarch)) {
		p = bpathf(&b, "%s/pkg/%s_%s", goroot, goos, goarch);
		if(rebuildall)
			xremoveall(p);
		xmkdirall(p);
	}

	// Create object directory.
	// We keep it in pkg/ so that all the generated binaries
	// are in one tree.  If pkg/obj/libgc.a exists, it is a dreg from
	// before we used subdirectories of obj.  Delete all of obj
	// to clean up.
	bpathf(&b, "%s/pkg/obj/libgc.a", goroot);
	if(isfile(bstr(&b)))
		xremoveall(bpathf(&b, "%s/pkg/obj", goroot));
	p = bpathf(&b, "%s/pkg/obj/%s_%s", goroot, gohostos, gohostarch);
	if(rebuildall)
		xremoveall(p);
	xmkdirall(p);

	// Create tool directory.
	// We keep it in pkg/, just like the object directory above.
	if(rebuildall)
		xremoveall(tooldir);
	xmkdirall(tooldir);

	// Remove tool binaries from before the tool/gohostos_gohostarch
	xremoveall(bpathf(&b, "%s/bin/tool", goroot));

	// Remove old pre-tool binaries.
	for(i=0; i<nelem(oldtool); i++)
		xremove(bpathf(&b, "%s/bin/%s", goroot, oldtool[i]));

	// If $GOBIN is set and has a Go compiler, it must be cleaned.
	for(i=0; gochars[i]; i++) {
		if(isfile(bprintf(&b, "%s%s%c%s", gobin, slash, gochars[i], "g"))) {
			for(i=0; i<nelem(oldtool); i++)
				xremove(bprintf(&b, "%s%s%s", gobin, slash, oldtool[i]));
			break;
		}
	}

	// For release, make sure excluded things are excluded.
	if(hasprefix(goversion, "release.") || (hasprefix(goversion, "go") && !contains(goversion, "beta"))) {
		for(i=0; i<nelem(unreleased); i++)
			if(isdir(bpathf(&b, "%s/%s", goroot, unreleased[i])))
				fatal("%s should not exist in release build", bstr(&b));
	}

	bfree(&b);
}

/*
 * C library and tool building
 */

// gccargs is the gcc command line to use for compiling a single C file.
static char *proto_gccargs[] = {
	"-Wall",
	// native Plan 9 compilers don't like non-standard prototypes
	// so let gcc catch them.
	"-Wstrict-prototypes",
	"-Wextra",
	"-Wunused",
	"-Wno-sign-compare",
	"-Wno-missing-braces",
	"-Wno-parentheses",
	"-Wno-unknown-pragmas",
	"-Wno-switch",
	"-Wno-comment",
	"-Wno-missing-field-initializers",
	"-Werror",
	"-fno-common",
	"-ggdb",
	"-pipe",
};

// gccargs2 is the second part of gccargs.
// it is used if the environment isn't defining CFLAGS.
static char *proto_gccargs2[] = {
	// on older versions of GCC, -Wuninitialized is not supported
	// without -O, so put it here together with -O settings in case
	// the user's $CFLAGS doesn't include -O.
	"-Wuninitialized",
#if defined(__NetBSD__) && defined(__arm__)
	// GCC 4.5.4 (NetBSD nb1 20120916) on ARM is known to mis-optimize gc/mparith3.c
	// Fix available at http://patchwork.ozlabs.org/patch/64562/.
	"-O1",
#else
	"-O2",
#endif
};

static Vec gccargs, ldargs;

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
	}},
	{"libbio", {
		"$GOROOT/include/u.h",
		"$GOROOT/include/utf.h",
		"$GOROOT/include/fmt.h",
		"$GOROOT/include/libc.h",
		"$GOROOT/include/bio.h",
	}},
	{"liblink", {
		"$GOROOT/include/u.h",
		"$GOROOT/include/utf.h",
		"$GOROOT/include/fmt.h",
		"$GOROOT/include/libc.h",
		"$GOROOT/include/bio.h",
		"$GOROOT/include/ar.h",
		"$GOROOT/include/link.h",
		"anames5.c",
		"anames6.c",
		"anames8.c",
	}},
	{"cmd/cc", {
		"-pgen.c",
		"-pswt.c",
	}},
	{"cmd/gc", {
		"-cplx.c",
		"-pgen.c",
		"-plive.c",
		"-popt.c",
		"-y1.tab.c",  // makefile dreg
		"opnames.h",
	}},
	{"cmd/5c", {
		"../cc/pgen.c",
		"../cc/pswt.c",
		"$GOROOT/pkg/obj/$GOHOSTOS_$GOHOSTARCH/libcc.a",
	}},
	{"cmd/6c", {
		"../cc/pgen.c",
		"../cc/pswt.c",
		"$GOROOT/pkg/obj/$GOHOSTOS_$GOHOSTARCH/libcc.a",
	}},
	{"cmd/8c", {
		"../cc/pgen.c",
		"../cc/pswt.c",
		"$GOROOT/pkg/obj/$GOHOSTOS_$GOHOSTARCH/libcc.a",
	}},
	{"cmd/5g", {
		"../gc/cplx.c",
		"../gc/pgen.c",
		"../gc/plive.c",
		"../gc/popt.c",
		"../gc/popt.h",
		"$GOROOT/pkg/obj/$GOHOSTOS_$GOHOSTARCH/libgc.a",
	}},
	{"cmd/6g", {
		"../gc/cplx.c",
		"../gc/pgen.c",
		"../gc/plive.c",
		"../gc/popt.c",
		"../gc/popt.h",
		"$GOROOT/pkg/obj/$GOHOSTOS_$GOHOSTARCH/libgc.a",
	}},
	{"cmd/8g", {
		"../gc/cplx.c",
		"../gc/pgen.c",
		"../gc/plive.c",
		"../gc/popt.c",
		"../gc/popt.h",
		"$GOROOT/pkg/obj/$GOHOSTOS_$GOHOSTARCH/libgc.a",
	}},
	{"cmd/5l", {
		"../ld/*",
	}},
	{"cmd/6l", {
		"../ld/*",
	}},
	{"cmd/8l", {
		"../ld/*",
	}},
	{"cmd/go", {
		"zdefaultcc.go",
	}},
	{"cmd/", {
		"$GOROOT/pkg/obj/$GOHOSTOS_$GOHOSTARCH/liblink.a",
		"$GOROOT/pkg/obj/$GOHOSTOS_$GOHOSTARCH/libbio.a",
		"$GOROOT/pkg/obj/$GOHOSTOS_$GOHOSTARCH/lib9.a",
	}},
	{"runtime", {
		"zaexperiment.h", // must sort above zasm
		"zasm_$GOOS_$GOARCH.h",
		"zsys_$GOOS_$GOARCH.s",
		"zgoarch_$GOARCH.go",
		"zgoos_$GOOS.go",
		"zruntime_defs_$GOOS_$GOARCH.go",
		"zversion.go",
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
	char *nameprefix;
	void (*gen)(char*, char*);
} gentab[] = {
	{"opnames.h", gcopnames},
	{"anames5.c", mkanames},
	{"anames6.c", mkanames},
	{"anames8.c", mkanames},
	{"zasm_", mkzasm},
	{"zdefaultcc.go", mkzdefaultcc},
	{"zsys_", mkzsys},
	{"zgoarch_", mkzgoarch},
	{"zgoos_", mkzgoos},
	{"zruntime_defs_", mkzruntimedefs},
	{"zversion.go", mkzversion},
	{"zaexperiment.h", mkzexperiment},

	// not generated anymore, but delete the file if we see it
	{"enam.c", nil},
};

// install installs the library, package, or binary associated with dir,
// which is relative to $GOROOT/src.
static void
install(char *dir)
{
	char *name, *p, *elem, *prefix, *exe;
	bool islib, ispkg, isgo, stale, ispackcmd;
	Buf b, b1, path, final_path, final_name;
	Vec compile, files, link, go, missing, clean, lib, extra;
	Time ttarg, t;
	int i, j, k, n, doclean, targ;

	if(vflag) {
		if(!streq(goos, gohostos) || !streq(goarch, gohostarch))
			errprintf("%s (%s/%s)\n", dir, goos, goarch);
		else
			errprintf("%s\n", dir);
	}

	binit(&b);
	binit(&b1);
	binit(&path);
	binit(&final_path);
	binit(&final_name);
	vinit(&compile);
	vinit(&files);
	vinit(&link);
	vinit(&go);
	vinit(&missing);
	vinit(&clean);
	vinit(&lib);
	vinit(&extra);


	// path = full path to dir.
	bpathf(&path, "%s/src/%s", goroot, dir);
	bpathf(&final_path, "%s/src/%s", goroot_final, dir);
	name = lastelem(dir);

	// set up gcc command line on first run.
	if(gccargs.len == 0) {
		bprintf(&b, "%s %s", defaultcc, defaultcflags);
		splitfields(&gccargs, bstr(&b));
		for(i=0; i<nelem(proto_gccargs); i++)
			vadd(&gccargs, proto_gccargs[i]);
		if(defaultcflags[0] == '\0') {
			for(i=0; i<nelem(proto_gccargs2); i++)
				vadd(&gccargs, proto_gccargs2[i]);
		}
		if(contains(gccargs.p[0], "clang")) {
			// disable ASCII art in clang errors, if possible
			vadd(&gccargs, "-fno-caret-diagnostics");
			// clang is too smart about unused command-line arguments
			vadd(&gccargs, "-Qunused-arguments");
		}
		// disable word wrapping in error messages
		vadd(&gccargs, "-fmessage-length=0");
		if(streq(gohostos, "darwin")) {
			// golang.org/issue/5261
			vadd(&gccargs, "-mmacosx-version-min=10.6");
		}
	}
	if(ldargs.len == 0 && defaultldflags[0] != '\0') {
		bprintf(&b, "%s", defaultldflags);
		splitfields(&ldargs, bstr(&b));
	}

	islib = hasprefix(dir, "lib") || streq(dir, "cmd/cc") || streq(dir, "cmd/gc");
	ispkg = !islib && !hasprefix(dir, "cmd/");
	isgo = ispkg || streq(dir, "cmd/go") || streq(dir, "cmd/cgo");

	exe = "";
	if(streq(gohostos, "windows"))
		exe = ".exe";

	// Start final link command line.
	// Note: code below knows that link.p[targ] is the target.
	ispackcmd = 0;
	if(islib) {
		// C library.
		vadd(&link, "ar");
		if(streq(gohostos, "plan9"))
			vadd(&link, "rc");
		else
			vadd(&link, "rsc");
		prefix = "";
		if(!hasprefix(name, "lib"))
			prefix = "lib";
		targ = link.len;
		vadd(&link, bpathf(&b, "%s/pkg/obj/%s_%s/%s%s.a", goroot, gohostos, gohostarch, prefix, name));
	} else if(ispkg) {
		// Go library (package).
		ispackcmd = 1;
		vadd(&link, "pack"); // program name - unused here, but all the other cases record one
		p = bprintf(&b, "%s/pkg/%s_%s/%s", goroot, goos, goarch, dir);
		*xstrrchr(p, '/') = '\0';
		xmkdirall(p);
		targ = link.len;
		vadd(&link, bpathf(&b, "%s/pkg/%s_%s/%s.a", goroot, goos, goarch, dir));
	} else if(streq(dir, "cmd/go") || streq(dir, "cmd/cgo")) {
		// Go command.
		vadd(&link, bpathf(&b, "%s/%sl", tooldir, gochar));
		vadd(&link, "-o");
		elem = name;
		if(streq(elem, "go"))
			elem = "go_bootstrap";
		targ = link.len;
		vadd(&link, bpathf(&b, "%s/%s%s", tooldir, elem, exe));
	} else {
		// C command. Use gccargs and ldargs.
		if(streq(gohostos, "plan9")) {
			vadd(&link, bprintf(&b, "%sl", gohostchar));
			vadd(&link, "-o");
			targ = link.len;
			vadd(&link, bpathf(&b, "%s/%s", tooldir, name));
		} else {
			vcopy(&link, gccargs.p, gccargs.len);
			vcopy(&link, ldargs.p, ldargs.len);
			if(sflag)
				vadd(&link, "-static");
			vadd(&link, "-o");
			targ = link.len;
			vadd(&link, bpathf(&b, "%s/%s%s", tooldir, name, exe));
			if(streq(gohostarch, "amd64"))
				vadd(&link, "-m64");
			else if(streq(gohostarch, "386"))
				vadd(&link, "-m32");
		}
	}
	ttarg = mtime(link.p[targ]);

	// Gather files that are sources for this target.
	// Everything in that directory, and any target-specific
	// additions.
	xreaddir(&files, bstr(&path));

	// Remove files beginning with . or _,
	// which are likely to be editor temporary files.
	// This is the same heuristic build.ScanDir uses.
	// There do exist real C files beginning with _,
	// so limit that check to just Go files.
	n = 0;
	for(i=0; i<files.len; i++) {
		p = files.p[i];
		if(hasprefix(p, ".") || (hasprefix(p, "_") && hassuffix(p, ".go")))
			xfree(p);
		else
			files.p[n++] = p;
	}
	files.len = n;

	for(i=0; i<nelem(deptab); i++) {
		if(streq(dir, deptab[i].prefix) ||
		   (hassuffix(deptab[i].prefix, "/") && hasprefix(dir, deptab[i].prefix))) {
			for(j=0; (p=deptab[i].dep[j])!=nil; j++) {
				breset(&b1);
				bwritestr(&b1, p);
				bsubst(&b1, "$GOROOT", goroot);
				bsubst(&b1, "$GOOS", goos);
				bsubst(&b1, "$GOARCH", goarch);
				bsubst(&b1, "$GOHOSTOS", gohostos);
				bsubst(&b1, "$GOHOSTARCH", gohostarch);
				p = bstr(&b1);
				if(hassuffix(p, ".a")) {
					vadd(&lib, bpathf(&b, "%s", p));
					continue;
				}
				if(hassuffix(p, "/*")) {
					bpathf(&b, "%s/%s", bstr(&path), p);
					b.len -= 2;
					xreaddir(&extra, bstr(&b));
					bprintf(&b, "%s", p);
					b.len -= 2;
					for(k=0; k<extra.len; k++)
						vadd(&files, bpathf(&b1, "%s/%s", bstr(&b), extra.p[k]));
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
			bpathf(&b, "%s/%s", bstr(&path), files.p[i]);
			xfree(files.p[i]);
			files.p[i] = btake(&b);
		}
	}

	// Is the target up-to-date?
	stale = rebuildall;
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
		if(t != 0 && !hassuffix(p, ".a") && !shouldbuild(p, dir)) {
			xfree(files.p[i]);
			continue;
		}
		if(hassuffix(p, ".go"))
			vadd(&go, p);
		if(t > ttarg)
			stale = 1;
		if(t == 0) {
			vadd(&missing, p);
			files.p[n++] = files.p[i];
			continue;
		}
		files.p[n++] = files.p[i];
	}
	files.len = n;

	// If there are no files to compile, we're done.
	if(files.len == 0)
		goto out;
	
	for(i=0; i<lib.len && !stale; i++)
		if(mtime(lib.p[i]) > ttarg)
			stale = 1;

	if(!stale)
		goto out;

	// For package runtime, copy some files into the work space.
	if(streq(dir, "runtime")) {
		copyfile(bpathf(&b, "%s/arch_GOARCH.h", workdir),
			bpathf(&b1, "%s/arch_%s.h", bstr(&path), goarch), 0);
		copyfile(bpathf(&b, "%s/defs_GOOS_GOARCH.h", workdir),
			bpathf(&b1, "%s/defs_%s_%s.h", bstr(&path), goos, goarch), 0);
		p = bpathf(&b1, "%s/signal_%s_%s.h", bstr(&path), goos, goarch);
		if(isfile(p))
			copyfile(bpathf(&b, "%s/signal_GOOS_GOARCH.h", workdir), p, 0);
		copyfile(bpathf(&b, "%s/os_GOOS.h", workdir),
			bpathf(&b1, "%s/os_%s.h", bstr(&path), goos), 0);
		copyfile(bpathf(&b, "%s/signals_GOOS.h", workdir),
			bpathf(&b1, "%s/signals_%s.h", bstr(&path), goos), 0);
		copyfile(bpathf(&b, "%s/pkg/%s_%s/textflag.h", goroot, goos, goarch),
			bpathf(&b1, "%s/src/cmd/ld/textflag.h", goroot), 0);
		copyfile(bpathf(&b, "%s/pkg/%s_%s/funcdata.h", goroot, goos, goarch),
			bpathf(&b1, "%s/src/runtime/funcdata.h", goroot), 0);
	}

	// Generate any missing files; regenerate existing ones.
	for(i=0; i<files.len; i++) {
		p = files.p[i];
		elem = lastelem(p);
		for(j=0; j<nelem(gentab); j++) {
			if(gentab[j].gen == nil)
				continue;
			if(hasprefix(elem, gentab[j].nameprefix)) {
				if(vflag > 1)
					errprintf("generate %s\n", p);
				gentab[j].gen(bstr(&path), p);
				// Do not add generated file to clean list.
				// In runtime, we want to be able to
				// build the package with the go tool,
				// and it assumes these generated files already
				// exist (it does not know how to build them).
				// The 'clean' command can remove
				// the generated files.
				goto built;
			}
		}
		// Did not rebuild p.
		if(find(p, missing.p, missing.len) >= 0)
			fatal("missing file %s", p);
	built:;
	}

	// One more copy for package runtime.
	// The last batch was required for the generators.
	// This one is generated.
	if(streq(dir, "runtime")) {
		copyfile(bpathf(&b, "%s/zasm_GOOS_GOARCH.h", workdir),
			bpathf(&b1, "%s/zasm_%s_%s.h", bstr(&path), goos, goarch), 0);
	}

	if((!streq(goos, gohostos) || !streq(goarch, gohostarch)) && isgo) {
		// We've generated the right files; the go command can do the build.
		if(vflag > 1)
			errprintf("skip build for cross-compile %s\n", dir);
		goto nobuild;
	}

	// Compile the files.
	for(i=0; i<files.len; i++) {
		if(!hassuffix(files.p[i], ".c") && !hassuffix(files.p[i], ".s"))
			continue;
		name = lastelem(files.p[i]);

		vreset(&compile);
		if(!isgo) {
			// C library or tool.
			if(streq(gohostos, "plan9")) {
				vadd(&compile, bprintf(&b, "%sc", gohostchar));
				vadd(&compile, "-FTVwp");
				vadd(&compile, "-DPLAN9");
				vadd(&compile, "-D__STDC__=1");
				vadd(&compile, "-D__SIZE_TYPE__=ulong"); // for GNU Bison
				vadd(&compile, bpathf(&b, "-I%s/include/plan9", goroot));
				vadd(&compile, bpathf(&b, "-I%s/include/plan9/%s", goroot, gohostarch));
			} else {
				vcopy(&compile, gccargs.p, gccargs.len);
				vadd(&compile, "-c");
				if(streq(gohostarch, "amd64"))
					vadd(&compile, "-m64");
				else if(streq(gohostarch, "386"))
					vadd(&compile, "-m32");
	
				vadd(&compile, "-I");
				vadd(&compile, bpathf(&b, "%s/include", goroot));
			}

			if(streq(dir, "lib9"))
				vadd(&compile, "-DPLAN9PORT");


			vadd(&compile, "-I");
			vadd(&compile, bstr(&path));

			// lib9/goos.c gets the default constants hard-coded.
			if(streq(name, "goos.c")) {
				vadd(&compile, "-D");
				vadd(&compile, bprintf(&b, "GOOS=\"%s\"", goos));
				vadd(&compile, "-D");
				vadd(&compile, bprintf(&b, "GOARCH=\"%s\"", goarch));
				bprintf(&b1, "%s", goroot_final);
				bsubst(&b1, "\\", "\\\\");  // turn into C string
				vadd(&compile, "-D");
				vadd(&compile, bprintf(&b, "GOROOT=\"%s\"", bstr(&b1)));
				vadd(&compile, "-D");
				vadd(&compile, bprintf(&b, "GOVERSION=\"%s\"", goversion));
				vadd(&compile, "-D");
				vadd(&compile, bprintf(&b, "GOARM=\"%s\"", goarm));
				vadd(&compile, "-D");
				vadd(&compile, bprintf(&b, "GO386=\"%s\"", go386));
				vadd(&compile, "-D");
				vadd(&compile, bprintf(&b, "GO_EXTLINK_ENABLED=\"%s\"", goextlinkenabled));
			}

			// gc/lex.c records the GOEXPERIMENT setting used during the build.
			if(streq(name, "lex.c")) {
				xgetenv(&b, "GOEXPERIMENT");
				vadd(&compile, "-D");
				vadd(&compile, bprintf(&b1, "GOEXPERIMENT=\"%s\"", bstr(&b)));
			}
		} else {
			// Supporting files for a Go package.
			if(hassuffix(files.p[i], ".s"))
				vadd(&compile, bpathf(&b, "%s/%sa", tooldir, gochar));
			else {
				vadd(&compile, bpathf(&b, "%s/%sc", tooldir, gochar));
				vadd(&compile, "-F");
				vadd(&compile, "-V");
				vadd(&compile, "-w");
			}
			vadd(&compile, "-I");
			vadd(&compile, workdir);
			vadd(&compile, "-I");
			vadd(&compile, bprintf(&b, "%s/pkg/%s_%s", goroot, goos, goarch));
			vadd(&compile, "-D");
			vadd(&compile, bprintf(&b, "GOOS_%s", goos));
			vadd(&compile, "-D");
			vadd(&compile, bprintf(&b, "GOARCH_%s", goarch));
			vadd(&compile, "-D");
			vadd(&compile, bprintf(&b, "GOOS_GOARCH_%s_%s", goos, goarch));
		}

		bpathf(&b, "%s/%s", workdir, lastelem(files.p[i]));
		doclean = 1;
		if(!isgo && streq(gohostos, "darwin")) {
			// To debug C programs on OS X, it is not enough to say -ggdb
			// on the command line.  You have to leave the object files
			// lying around too.  Leave them in pkg/obj/, which does not
			// get removed when this tool exits.
			bpathf(&b1, "%s/pkg/obj/%s", goroot, dir);
			xmkdirall(bstr(&b1));
			bpathf(&b, "%s/%s", bstr(&b1), lastelem(files.p[i]));
			doclean = 0;
		}

		// Change the last character of the output file (which was c or s).
		if(streq(gohostos, "plan9"))
			b.p[b.len-1] = gohostchar[0];
		else
			b.p[b.len-1] = 'o';
		vadd(&compile, "-o");
		vadd(&compile, bstr(&b));
		vadd(&compile, files.p[i]);
		bgrunv(bstr(&path), CheckExit, &compile);

		vadd(&link, bstr(&b));
		if(doclean)
			vadd(&clean, bstr(&b));
	}
	bgwait();

	if(isgo) {
		// The last loop was compiling individual files.
		// Hand the Go files to the compiler en masse.
		vreset(&compile);
		vadd(&compile, bpathf(&b, "%s/%sg", tooldir, gochar));

		bpathf(&b, "%s/_go_.a", workdir);
		vadd(&compile, "-pack");
		vadd(&compile, "-o");
		vadd(&compile, bstr(&b));
		vadd(&clean, bstr(&b));
		if(!ispackcmd)
			vadd(&link, bstr(&b));

		vadd(&compile, "-p");
		if(hasprefix(dir, "pkg/"))
			vadd(&compile, dir+4);
		else
			vadd(&compile, "main");

		if(streq(dir, "runtime"))
			vadd(&compile, "-+");

		vcopy(&compile, go.p, go.len);

		runv(nil, bstr(&path), CheckExit, &compile);

		if(ispackcmd) {
			xremove(link.p[targ]);
			dopack(link.p[targ], bstr(&b), &link.p[targ+1], link.len - (targ+1));
			goto nobuild;
		}
	}

	if(!islib && !isgo) {
		// C binaries need the libraries explicitly, and -lm.
		vcopy(&link, lib.p, lib.len);
		if(!streq(gohostos, "plan9"))
			vadd(&link, "-lm");
	}

	// Remove target before writing it.
	xremove(link.p[targ]);

	runv(nil, nil, CheckExit, &link);

nobuild:
	// In package runtime, we install runtime.h and cgocall.h too,
	// for use by cgo compilation.
	if(streq(dir, "runtime")) {
		copyfile(bpathf(&b, "%s/pkg/%s_%s/cgocall.h", goroot, goos, goarch),
			bpathf(&b1, "%s/src/runtime/cgocall.h", goroot), 0);
		copyfile(bpathf(&b, "%s/pkg/%s_%s/runtime.h", goroot, goos, goarch),
			bpathf(&b1, "%s/src/runtime/runtime.h", goroot), 0);
	}


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
	char *p;
	bool res;

	p = xstrrchr(f, ',');
	if(p == nil)
		return streq(f, goos) || streq(f, goarch) || streq(f, "cmd_go_bootstrap") || streq(f, "go1.1") || (streq(goos, "android") && streq(f, "linux"));
	*p = 0;
	res = matchfield(f) && matchfield(p+1);
	*p = ',';
	return res;
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
	int i, j, ret;
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

	// cmd/go/doc.go has a giant /* */ comment before
	// it gets to the important detail that it is not part of
	// package main.  We don't parse those comments,
	// so special case that file.
	if(hassuffix(file, "cmd/go/doc.go") || hassuffix(file, "cmd\\go\\doc.go"))
		return 0;
	if(hassuffix(file, "cmd/cgo/doc.go") || hassuffix(file, "cmd\\cgo\\doc.go"))
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
		if(contains(p, "package main") && !streq(dir, "cmd/go") && !streq(dir, "cmd/cgo")) {
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

// copy copies the file src to dst, via memory (so only good for small files).
void
copyfile(char *dst, char *src, int exec)
{
	Buf b;

	if(vflag > 1)
		errprintf("cp %s %s\n", src, dst);

	binit(&b);
	readfile(&b, src);
	writefile(&b, dst, exec);
	bfree(&b);
}

// dopack copies the package src to dst,
// appending the files listed in extra.
// The archive format is the traditional Unix ar format.
static void
dopack(char *dst, char *src, char **extra, int nextra)
{
	int i;
	char c, *p, *q;
	Buf b, bdst;
	
	binit(&b);
	binit(&bdst);

	readfile(&bdst, src);
	for(i=0; i<nextra; i++) {
		readfile(&b, extra[i]);
		// find last path element for archive member name
		p = xstrrchr(extra[i], '/');
		if(p)
			p++;
		q = xstrrchr(extra[i], '\\');
		if(q) {
			q++;
			if(p == nil || q > p)
				p = q;
		}
		if(p == nil)
			p = extra[i];
		bwritef(&bdst, "%-16.16s%-12d%-6d%-6d%-8o%-10d`\n", p, 0, 0, 0, 0644, b.len);
		bwriteb(&bdst, &b);
		if(b.len&1) {
			c = 0;
			bwrite(&bdst, &c, 1);
		}
	}

	writefile(&bdst, dst, 0);

	bfree(&b);
	bfree(&bdst);
}

// buildorder records the order of builds for the 'go bootstrap' command.
static char *buildorder[] = {
	"lib9",
	"libbio",
	"liblink",

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
	"runtime",
	"errors",
	"sync/atomic",
	"sync",
	"io",
	"unicode",
	"unicode/utf8",
	"unicode/utf16",
	"bytes",
	"math",
	"strings",
	"strconv",
	"bufio",
	"sort",
	"container/heap",
	"encoding/base64",
	"syscall",
	"time",
	"os",
	"reflect",
	"fmt",
	"encoding",
	"encoding/json",
	"flag",
	"path/filepath",
	"path",
	"io/ioutil",
	"log",
	"regexp/syntax",
	"regexp",
	"go/token",
	"go/scanner",
	"go/ast",
	"go/parser",
	"os/exec",
	"os/signal",
	"net/url",
	"text/template/parse",
	"text/template",
	"go/doc",
	"go/build",
	"cmd/go",
};

// cleantab records the directories to clean in 'go clean'.
// It is bigger than the buildorder because we clean all the
// compilers but build only the $GOARCH ones.
static char *cleantab[] = {
	// Commands and C libraries.
	"cmd/5a",
	"cmd/5c",
	"cmd/5g",
	"cmd/5l",
	"cmd/6a",
	"cmd/6c",
	"cmd/6g",
	"cmd/6l",
	"cmd/8a",
	"cmd/8c",
	"cmd/8g",
	"cmd/8l",
	"cmd/cc",
	"cmd/gc",
	"cmd/go",	
	"lib9",
	"libbio",
	"liblink",

	// Go packages.
	"bufio",
	"bytes",
	"container/heap",
	"encoding",
	"encoding/base64",
	"encoding/json",
	"errors",
	"flag",
	"fmt",
	"go/ast",
	"go/build",
	"go/doc",
	"go/parser",
	"go/scanner",
	"go/token",
	"io",
	"io/ioutil",
	"log",
	"math",
	"net/url",
	"os",
	"os/exec",
	"path",
	"path/filepath",
	"reflect",
	"regexp",
	"regexp/syntax",
	"runtime",
	"sort",
	"strconv",
	"strings",
	"sync",
	"sync/atomic",
	"syscall",
	"text/template",
	"text/template/parse",
	"time",
	"unicode",
	"unicode/utf16",
	"unicode/utf8",
};

static void
clean(void)
{
	int i, j, k;
	Buf b, path;
	Vec dir;

	binit(&b);
	binit(&path);
	vinit(&dir);

	for(i=0; i<nelem(cleantab); i++) {
		bpathf(&path, "%s/src/%s", goroot, cleantab[i]);
		xreaddir(&dir, bstr(&path));
		// Remove generated files.
		for(j=0; j<dir.len; j++) {
			for(k=0; k<nelem(gentab); k++) {
				if(hasprefix(dir.p[j], gentab[k].nameprefix))
					xremove(bpathf(&b, "%s/%s", bstr(&path), dir.p[j]));
			}
		}
		// Remove generated binary named for directory.
		if(hasprefix(cleantab[i], "cmd/"))
			xremove(bpathf(&b, "%s/%s", bstr(&path), cleantab[i]+4));
	}

	// remove src/runtime/z* unconditionally
	vreset(&dir);
	bpathf(&path, "%s/src/runtime", goroot);
	xreaddir(&dir, bstr(&path));
	for(j=0; j<dir.len; j++) {
		if(hasprefix(dir.p[j], "z"))
			xremove(bpathf(&b, "%s/%s", bstr(&path), dir.p[j]));
	}

	if(rebuildall) {
		// Remove object tree.
		xremoveall(bpathf(&b, "%s/pkg/obj/%s_%s", goroot, gohostos, gohostarch));

		// Remove installed packages and tools.
		xremoveall(bpathf(&b, "%s/pkg/%s_%s", goroot, gohostos, gohostarch));
		xremoveall(bpathf(&b, "%s/pkg/%s_%s", goroot, goos, goarch));
		xremoveall(tooldir);

		// Remove cached version info.
		xremove(bpathf(&b, "%s/VERSION.cache", goroot));
	}

	bfree(&b);
	bfree(&path);
	vfree(&dir);
}

/*
 * command implementations
 */

void
usage(void)
{
	xprintf("usage: go tool dist [command]\n"
		"Commands are:\n"
		"\n"
		"banner         print installation banner\n"
		"bootstrap      rebuild everything\n"
		"clean          deletes all built files\n"
		"env [-p]       print environment (-p: include $PATH)\n"
		"install [dir]  install individual directory\n"
		"version        print Go version\n"
		"\n"
		"All commands take -v flags to emit extra information.\n"
	);
	xexit(2);
}

// The env command prints the default environment.
void
cmdenv(int argc, char **argv)
{
	bool pflag;
	char *sep;
	Buf b, b1;
	char *format;

	binit(&b);
	binit(&b1);

	format = "%s=\"%s\"\n";
	pflag = 0;
	ARGBEGIN{
	case '9':
		format = "%s='%s'\n";
		break;
	case 'p':
		pflag = 1;
		break;
	case 'v':
		vflag++;
		break;
	case 'w':
		format = "set %s=%s\r\n";
		break;
	default:
		usage();
	}ARGEND

	if(argc > 0)
		usage();

	xprintf(format, "CC", defaultcc);
	xprintf(format, "CC_FOR_TARGET", defaultcctarget);
	xprintf(format, "GOROOT", goroot);
	xprintf(format, "GOBIN", gobin);
	xprintf(format, "GOARCH", goarch);
	xprintf(format, "GOOS", goos);
	xprintf(format, "GOHOSTARCH", gohostarch);
	xprintf(format, "GOHOSTOS", gohostos);
	xprintf(format, "GOTOOLDIR", tooldir);
	xprintf(format, "GOCHAR", gochar);
	if(streq(goarch, "arm"))
		xprintf(format, "GOARM", goarm);
	if(streq(goarch, "386"))
		xprintf(format, "GO386", go386);

	if(pflag) {
		sep = ":";
		if(streq(gohostos, "windows"))
			sep = ";";
		xgetenv(&b, "PATH");
		bprintf(&b1, "%s%s%s", gobin, sep, bstr(&b));
		xprintf(format, "PATH", bstr(&b1));
	}

	bfree(&b);
	bfree(&b1);
}

// The bootstrap command runs a build from scratch,
// stopping at having installed the go_bootstrap command.
void
cmdbootstrap(int argc, char **argv)
{
	int i;
	Buf b;
	char *oldgoos, *oldgoarch, *oldgochar;

	binit(&b);

	ARGBEGIN{
	case 'a':
		rebuildall = 1;
		break;
	case 's':
		sflag++;
		break;
	case 'v':
		vflag++;
		break;
	default:
		usage();
	}ARGEND

	if(argc > 0)
		usage();

	if(isdir(bpathf(&b, "%s/src/pkg", goroot))) {
		fatal("\n\n"
			"The Go package sources have moved to $GOROOT/src.\n"
			"*** %s still exists. ***\n"
			"It probably contains stale files that may confuse the build.\n"
			"Please (check what's there and) remove it and try again.\n"
			"See http://golang.org/s/go14nopkg\n", bpathf(&b, "%s/src/pkg", goroot));
	}
	
	if(rebuildall)
		clean();
	goversion = findgoversion();
	setup();

	xsetenv("GOROOT", goroot);
	xsetenv("GOROOT_FINAL", goroot_final);

	// For the main bootstrap, building for host os/arch.
	oldgoos = goos;
	oldgoarch = goarch;
	oldgochar = gochar;
	goos = gohostos;
	goarch = gohostarch;
	gochar = gohostchar;
	xsetenv("GOARCH", goarch);
	xsetenv("GOOS", goos);

	for(i=0; i<nelem(buildorder); i++) {
		install(bprintf(&b, buildorder[i], gohostchar));
		if(!streq(oldgochar, gohostchar) && xstrstr(buildorder[i], "%s"))
			install(bprintf(&b, buildorder[i], oldgochar));
	}

	goos = oldgoos;
	goarch = oldgoarch;
	gochar = oldgochar;
	xsetenv("GOARCH", goarch);
	xsetenv("GOOS", goos);

	// Build runtime for actual goos/goarch too.
	if(!streq(goos, gohostos) || !streq(goarch, gohostarch))
		install("runtime");

	bfree(&b);
}

static char*
defaulttarg(void)
{
	char *p;
	Buf pwd, src, real_src;

	binit(&pwd);
	binit(&src);
	binit(&real_src);

	// xgetwd might return a path with symlinks fully resolved, and if
	// there happens to be symlinks in goroot, then the hasprefix test
	// will never succeed. Instead, we use xrealwd to get a canonical
	// goroot/src before the comparison to avoid this problem.
	xgetwd(&pwd);
	p = btake(&pwd);
	bpathf(&src, "%s/src/", goroot);
	xrealwd(&real_src, bstr(&src));
	if(!hasprefix(p, bstr(&real_src)))
		fatal("current directory %s is not under %s", p, bstr(&real_src));
	p += real_src.len;
	// guard againt xrealwd return the directory without the trailing /
	if(*p == slash[0])
		p++;

	bfree(&pwd);
	bfree(&src);
	bfree(&real_src);

	return p;
}

// Install installs the list of packages named on the command line.
void
cmdinstall(int argc, char **argv)
{
	int i;

	ARGBEGIN{
	case 's':
		sflag++;
		break;
	case 'v':
		vflag++;
		break;
	default:
		usage();
	}ARGEND

	if(argc == 0)
		install(defaulttarg());

	for(i=0; i<argc; i++)
		install(argv[i]);
}

// Clean deletes temporary objects.
// Clean -i deletes the installed objects too.
void
cmdclean(int argc, char **argv)
{
	ARGBEGIN{
	case 'v':
		vflag++;
		break;
	default:
		usage();
	}ARGEND

	if(argc > 0)
		usage();

	clean();
}

// Banner prints the 'now you've installed Go' banner.
void
cmdbanner(int argc, char **argv)
{
	char *pathsep, *pid, *ns;
	Buf b, b1, search, path;

	ARGBEGIN{
	case 'v':
		vflag++;
		break;
	default:
		usage();
	}ARGEND

	if(argc > 0)
		usage();

	binit(&b);
	binit(&b1);
	binit(&search);
	binit(&path);

	xprintf("\n");
	xprintf("---\n");
	xprintf("Installed Go for %s/%s in %s\n", goos, goarch, goroot);
	xprintf("Installed commands in %s\n", gobin);

	if(!xsamefile(goroot_final, goroot)) {
		// If the files are to be moved, don't check that gobin
		// is on PATH; assume they know what they are doing.
	} else if(streq(gohostos, "plan9")) {
		// Check that gobin is bound before /bin.
		readfile(&b, "#c/pid");
		bsubst(&b, " ", "");
		pid = btake(&b);
		bprintf(&b, "/proc/%s/ns", pid);
		ns = btake(&b);
		readfile(&b, ns);
		bprintf(&search, "bind -b %s /bin\n", gobin);
		if(xstrstr(bstr(&b), bstr(&search)) == nil)
			xprintf("*** You need to bind %s before /bin.\n", gobin);
	} else {
		// Check that gobin appears in $PATH.
		xgetenv(&b, "PATH");
		pathsep = ":";
		if(streq(gohostos, "windows"))
			pathsep = ";";
		bprintf(&b1, "%s%s%s", pathsep, bstr(&b), pathsep);
		bprintf(&search, "%s%s%s", pathsep, gobin, pathsep);
		if(xstrstr(bstr(&b1), bstr(&search)) == nil)
			xprintf("*** You need to add %s to your PATH.\n", gobin);
	}

	if(streq(gohostos, "darwin")) {
		if(isfile(bpathf(&path, "%s/cov", tooldir)))
			xprintf("\n"
				"On OS X the debuggers must be installed setgid procmod.\n"
				"Read and run ./sudo.bash to install the debuggers.\n");
	}

	if(!xsamefile(goroot_final, goroot)) {
		xprintf("\n"
			"The binaries expect %s to be copied or moved to %s\n",
			goroot, goroot_final);
	}

	bfree(&b);
	bfree(&b1);
	bfree(&search);
	bfree(&path);
}

// Version prints the Go version.
void
cmdversion(int argc, char **argv)
{
	ARGBEGIN{
	case 'v':
		vflag++;
		break;
	default:
		usage();
	}ARGEND

	if(argc > 0)
		usage();

	xprintf("%s\n", goversion);
}
