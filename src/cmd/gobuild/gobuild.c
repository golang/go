// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Build a collection of go programs into a single package.

#include <u.h>
#include <unistd.h>
#include <libc.h>
#include <bio.h>

void
usage(void)
{
	fprint(2, "usage: gobuild [-m] packagename *.go *.c *.s\n");
	exits("usage");
}

int chatty;
int devnull;	// fd of /dev/null
int makefile;	// generate Makefile
char *thechar;	// object character
char *goos;
char *goarch;

// Info about when to compile a particular file.
typedef struct Job Job;
struct Job
{
	char *name;
	int pass;
};

// Run the command in argv.
// Return -1 if it fails (non-zero exit status).
// Return 0 on success.
// Showoutput controls whether to let output from command display
// on standard output and standard error.
int
run(char **argv, int showoutput)
{
	int pid, i;
	Waitmsg *w;
	vlong n0, n1;

	n0 = nsec();
	pid = fork();
	if(pid < 0)
		sysfatal("fork: %r");
	if(pid == 0){
		dup(devnull, 0);
		if(!showoutput){
			dup(devnull, 1);
			dup(devnull, 2);
		}else{
			dup(2, 1);
		}
		if(devnull > 2)
			close(devnull);
		exec(argv[0], argv);
		fprint(2, "exec %s: %r\n", argv[0]);
		exit(1);
	}
	w = waitfor(pid);
	n1 = nsec();
	if(w == nil)
		sysfatal("waitfor %d: %r", pid);
	if(chatty > 1){
		fprint(2, "%5.3f", (n1-n0)/1.e9);
		for(i=0; argv[i]; i++)
			fprint(2, " %s", argv[i]);
		if(w->msg[0])
			fprint(2, " [%s]", w->msg);
		fprint(2, "\n");
	}
	if(w->msg[0])
		return -1;
	return 0;
}

// Build the file using the compiler cc.
// Return -1 on error, 0 on success.
// If show is set, print the command and the output.
int
buildcc(char *cc, char *file, int show)
{
	char *argv[3];

	if(show)
		fprint(2, "$ %s %s\n", cc, file);
	argv[0] = cc;
	argv[1] = file;
	argv[2] = nil;
	return run(argv, show);
}

// Return bool whether s ends in suffix.
int
suffix(char *s, char *suffix)
{
	int n1, n2;

	n1 = strlen(s);
	n2 = strlen(suffix);
	if(n1>n2 && strcmp(s+n1-n2, suffix) == 0)
		return 1;
	return 0;
}

// Return the name of the compiler for file.
char*
compiler(char *file)
{
	static char buf[20];

	if(suffix(file, ".go"))
		snprint(buf, sizeof buf, "%sg", thechar);
	else if(suffix(file, ".c"))
		snprint(buf, sizeof buf, "%sc", thechar);
	else if(suffix(file, ".s"))
		snprint(buf, sizeof buf, "%sa", thechar);
	else
		sysfatal("don't know how to build %s", file);
	return buf;
}

// Return the object name for file, replacing the
// .c or .g or .a with .suffix.
char*
goobj(char *file, char *suffix)
{
	char *p;

	p = strrchr(file, '.');
	if(p == nil)
		sysfatal("don't know object name for %s", file);
	return smprint("%.*s.%s", utfnlen(file, p-file), file, suffix);
}

// Format name using $(GOOS) and $(GOARCH).
int
dollarfmt(Fmt *f)
{
	char *s;
	Rune r;
	int n;

	s = va_arg(f->args, char*);
	if(s == nil){
		fmtstrcpy(f, "<nil>");
		return 0;
	}
	for(; *s; s+=n){
		n = strlen(goarch);
		if(strncmp(s, goarch, n) == 0){
			fmtstrcpy(f, "$(GOARCH)");
			continue;
		}
		n = strlen(goos);
		if(strncmp(s, goos, n) == 0){
			fmtstrcpy(f, "$(GOOS)");
			continue;
		}
		n = chartorune(&r, s);
		fmtrune(f, r);
	}
	return 0;
}

// Makefile preamble template.
char preamble[] =
	"O=%s\n"
	"GC=$(O)g\n"
	"CC=$(O)c -w\n"
	"AS=$(O)a\n"
	"AR=$(O)ar\n"
	"\n"
	"PKG=%s.a\n"
	"PKGDIR=$(GOROOT)/pkg%s\n"
	"\n"
	"install: $(PKG)\n"
	"\tmv $(PKG) $(PKGDIR)/$(PKG)\n"
	"\n"
	"nuke: clean\n"
	"\trm -f $(PKGDIR)/$(PKG)\n"
	"\n"
	"clean:\n"
	"\trm -f *.$O *.a $(PKG)\n"
	"\n"
	"%%.$O: %%.go\n"
	"\t$(GC) $*.go\n"
	"\n"
	"%%.$O: %%.c\n"
	"\t$(CC) $*.c\n"
	"\n"
	"%%.$O: %%.s\n"
	"\t$(AS) $*.s\n"
	"\n"
;

void
main(int argc, char **argv)
{
	int i, o, p, n, pass, nar, njob, nthis, nnext, oargc;
	char **ar, **next, **this, **tmp, *goroot, *pkgname, *pkgpath, *pkgdir, **oargv, *q;
	Job *job;
	Biobuf bout;

	oargc = argc;
	oargv = argv;
	fmtinstall('$', dollarfmt);

	ARGBEGIN{
	default:
		usage();
	case 'm':
		makefile = 1;
		break;
	case 'v':
		chatty++;
		break;
	}ARGEND

	if(argc < 2)
		usage();

	goos = getenv("GOOS");
	if(goos == nil)
		sysfatal("no $GOOS");
	goarch = getenv("GOARCH");
	if(goarch == nil)
		sysfatal("no $GOARCH");
	if(strcmp(goarch, "amd64") == 0)
		thechar = "6";
	else
		sysfatal("unknown $GOARCH");

	goroot = getenv("GOROOT");
	if(goroot == nil)
		sysfatal("no $GOROOT");

	pkgname = argv[0];
	if(strchr(pkgname, '.')){
		fprint(2, "pkgname has dot\n");
		usage();
	}

	q = strrchr(pkgname, '/');
	if(q) {
		pkgdir = pkgname;
		*q++ = '\0';
		pkgname = q;
		pkgdir = smprint("/%s", pkgdir);
	} else {
		pkgdir = "";
	}

	pkgpath = smprint("%s.a", pkgname);
	unlink(pkgpath);
	if(chatty)
		fprint(2, "pkg %s\n", pkgpath);

	if((devnull = open("/dev/null", ORDWR)) < 0)
		sysfatal("open /dev/null: %r");

	// Compile by repeated passes: build as many .6 as you can,
	// put them all in the archive, and repeat.
	//
	// "this" contains the list of files to compile in this pass.
	// "next" contains the list of files to re-try in the next pass.
	// "job" contains the list of files that are done, annotated
	//	with their pass numbers.
	// "ar" contains the ar command line to run at the end
	//	of the pass.

	n = argc-1;
	this = malloc(n*sizeof this[0]);
	next = malloc(n*sizeof next[0]);
	job = malloc(n*sizeof job[0]);
	ar = malloc((n+4)*sizeof job[0]);
	if(this == nil || next == nil || job == 0 || ar == 0)
		sysfatal("malloc: %r");

	// Initial "this" is the files given on the command line.
	for(i=0; i<n; i++)
		this[i] = argv[i+1];
	nthis = n;

	ar[0] = smprint("%sar", thechar);
	ar[1] = "grc";
	ar[2] = pkgpath;
	ar[3] = nil;
	if(run(ar, 1) < 0)
		sysfatal("ar: %r");

	njob = 0;

	for(pass=0; nthis > 0; pass++){
		nnext = 0;
		nar = 3;

		// Try to build.
		for(i=0; i<nthis; i++){
			if(buildcc(compiler(this[i]), this[i], 0) < 0){
				next[nnext++] = this[i];
			}else{
				job[njob].pass = pass;
				job[njob++].name = this[i];
				ar[nar++] = goobj(this[i], thechar);
				if(chatty == 1)
					fprint(2, "%s ", this[i]);
			}
		}
		if(nthis == nnext){	// they all failed
			fprint(2, "cannot make progress\n");
			for(i=0; i<nthis; i++)
				buildcc(compiler(this[i]), this[i], 1);
			exits("stalemate");
		}
		if(chatty == 1)
			fprint(2, "\n");

		// Add to archive.
		ar[nar] = nil;
		if(run(ar, 1) < 0)
			sysfatal("ar: %r");

		// Delete objects.
		for(i=3; i<nar; i++)
			unlink(ar[i]);

		// Set up for next pass: next = this.
		tmp = next;
		next = this;
		this = tmp;
		nthis = nnext;
	}

	if(makefile){
		// Write makefile.
		Binit(&bout, 1, OWRITE);
		Bprint(&bout, "# DO NOT EDIT.  Automatically generated by gobuild.\n");
		o = Boffset(&bout);
		Bprint(&bout, "#");
		for(i=0; i<oargc; i++){
			if(Boffset(&bout) - o > 60){
				Bprint(&bout, "\\\n#   ");
				o = Boffset(&bout);
			}
			Bprint(&bout, " %s", oargv[i]);
		}
		Bprint(&bout, "\n");
		Bprint(&bout, preamble, thechar, pkgname, pkgdir);

		// O2=\
		//	os_file.$O\
		//	os_time.$O\
		//
		p = -1;
		for(i=0; i<n; i++){
			if(job[i].pass != p){
				p = job[i].pass;
				Bprint(&bout, "\nO%d=\\\n", p+1);
			}
			Bprint(&bout, "\t%$\\\n", goobj(job[i].name, "$O"));
		}
		Bprint(&bout, "\n");

		// $(PKG): a1 a2
		Bprint(&bout, "$(PKG):");
		for(i=0; i<pass; i++)
			Bprint(&bout, " a%d", i+1);
		Bprint(&bout, "\n");

		// a1: $(O1)
		//	$(AS) grc $(PKG) $(O1)
		//	rm -f $(O1)
		for(i=0; i<pass; i++){
			Bprint(&bout, "a%d:\t$(O%d)\n", i+1, i+1);
			Bprint(&bout, "\t$(AR) grc $(PKG) $(O%d)\n", i+1);
			Bprint(&bout, "\trm -f $(O%d)\n", i+1);
		}
		Bprint(&bout, "\n");

		// $(O1): nuke
		// $(O2): a1
		Bprint(&bout, "$(O1): nuke\n");
		for(i=1; i<pass; i++)
			Bprint(&bout, "$(O%d): a%d\n", i+1, i);
		Bprint(&bout, "\n");
		Bterm(&bout);
	}

	exits(0);
}
