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
	fprint(2, "usage: gobuild [-m] [packagename...]\n");
	exits("usage");
}

int chatty;
int devnull;	// fd of /dev/null
int makefile;	// generate Makefile
char *thechar;	// object character
char *goos;
char *goarch;
char *goroot;
char **oargv;
int oargc;

void writemakefile(void);
int sourcefilenames(char***);

void*
emalloc(int n)
{
	void *v;

	v = malloc(n);
	if(v == nil)
		sysfatal("out of memory");
	memset(v, 0, n);
	return v;
}

void*
erealloc(void *v, int n)
{
	v = realloc(v, n);
	if(v == nil)
		sysfatal("out of memory");
	return v;
}

// Info about when to compile a particular file.
typedef struct Job Job;
struct Job
{
	char *name;
	char *pkg;
	int pass;
};
Job *job;
int njob;

char **pkg;
int npkg;

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
	char buf[100];

	n0 = nsec();
	pid = fork();
	if(pid < 0)
		sysfatal("fork: %r");
	if(pid == 0){
		dup(devnull, 0);
		if(!showoutput)
			dup(devnull, 2);
		dup(2, 1);
		if(devnull > 2)
			close(devnull);
		exec(argv[0], argv);
		fprint(2, "exec %s: %r\n", argv[0]);
		exit(1);
	}
	while((w = waitfor(pid)) == nil) {
		rerrstr(buf, sizeof buf);
		if(strstr(buf, "interrupt"))
			continue;
		sysfatal("waitfor %d: %r", pid);
	}
	n1 = nsec();
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

// Run ar to add the given files to pkg.a.
void
ar(char *pkg, char **file, int nfile)
{
	char **arg;
	int i, n;
	char sixar[20];
	char pkga[1000];

	arg = emalloc((4+nfile)*sizeof arg[0]);
	n = 0;
	snprint(sixar, sizeof sixar, "%sar", thechar);
	snprint(pkga, sizeof pkga, "%s.a", pkg);
	arg[n++] = sixar;
	arg[n++] = "grc";
	arg[n++] = pkga;
	for(i=0; i<nfile; i++)
		arg[n++] = file[i];
	arg[n] = nil;

	if(run(arg, 1) < 0)
		sysfatal("ar: %r");
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

// Figure out package of .go file.
// Maintain list of all packages seen so far.
// Returned package string is in that list,
// so caller can use pointer compares.
char*
getpkg(char *file)
{
	Biobuf *b;
	char *p, *q;
	int i;

	if((b = Bopen(file, OREAD)) == nil)
		sysfatal("open %s: %r", file);
	while((p = Brdline(b, '\n')) != nil) {
		p[Blinelen(b)-1] = '\0';
		if(!suffix(file, ".go")) {
			if(*p != '/' || *(p+1) != '/')
				continue;
			p += 2;
		}
		if(strstr(p, "gobuild: ignore"))
			return "main";
		while(*p == ' ' || *p == '\t')
			p++;
		if(strncmp(p, "package", 7) == 0 && (p[7] == ' ' || p[7] == '\t')) {
			p+=7;
			while(*p == ' ' || *p == '\t')
				p++;
			q = p+strlen(p);
			while(q > p && (*(q-1) == ' ' || *(q-1) == '\t'))
				*--q = '\0';
			for(i=0; i<npkg; i++) {
				if(strcmp(pkg[i], p) == 0) {
					Bterm(b);
					return pkg[i];
				}
			}
			// don't put main in the package list
			if(strcmp(p, "main") == 0)
				return "main";
			npkg++;
			pkg = erealloc(pkg, npkg*sizeof pkg[0]);
			pkg[i] = emalloc(strlen(p)+1);
			strcpy(pkg[i], p);
			Bterm(b);
			return pkg[i];
		}
	}
	Bterm(b);
	return nil;
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
			if(f->flags & FmtSharp)
				fmtstrcpy(f, "${GOARCH}");  // shell
			else
				fmtstrcpy(f, "$(GOARCH)");  // make
			continue;
		}
		n = strlen(goos);
		if(strncmp(s, goos, n) == 0){
			if(f->flags & FmtSharp)
				fmtstrcpy(f, "${GOOS}");  // shell
			else
				fmtstrcpy(f, "$(GOOS)");  // make
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
	"default: packages\n"
	"\n"
	"clean:\n"
	"\trm -f *.$O *.a $O.out\n"
	"\n"
	"test: packages\n"
	"\tgotest\n"
	"\n"
	"coverage: packages\n"
	"\tgotest\n"
	"\t6cov -g `pwd` | grep -v '_test\\.go:'\n"
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
writemakefile(void)
{
	Biobuf bout;
	vlong o;
	int i, k, l, pass;
	char **obj;
	int nobj;

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
		Bprint(&bout, " %#$", oargv[i]);
	}
	Bprint(&bout, " >Makefile\n");
	Bprint(&bout, preamble, thechar);

	// O2=\
	//	os_file.$O\
	//	os_time.$O\
	//
	obj = emalloc(njob*sizeof obj[0]);
	for(pass=0;; pass++) {
		nobj = 0;
		for(i=0; i<njob; i++)
			if(job[i].pass == pass)
				obj[nobj++] = goobj(job[i].name, "$O");
		if(nobj == 0)
			break;
		Bprint(&bout, "O%d=\\\n", pass+1);
		for(i=0; i<nobj; i++)
			Bprint(&bout, "\t%$\\\n", obj[i]);
		Bprint(&bout, "\n");
	}

	// math.a: a1 a2
	for(i=0; i<npkg; i++) {
		Bprint(&bout, "%s.a:", pkg[i]);
		for(k=0; k<pass; k++)
			Bprint(&bout, " a%d", k+1);
		Bprint(&bout, "\n");
	}
	Bprint(&bout, "\n");

	// a1: $(O1)
	//	$(AS) grc $(PKG) $(O1)
	//	rm -f $(O1)
	for(k=0; k<pass; k++){
		Bprint(&bout, "a%d:\t$(O%d)\n", k+1, k+1);
		for(i=0; i<npkg; i++) {
			nobj = 0;
			for(l=0; l<njob; l++)
				if(job[l].pass == k && job[l].pkg == pkg[i])
					obj[nobj++] = goobj(job[l].name, "$O");
			if(nobj > 0) {
				Bprint(&bout, "\t$(AR) grc %s.a", pkg[i]);
				for(l=0; l<nobj; l++)
					Bprint(&bout, " %$", obj[l]);
				Bprint(&bout, "\n");
			}
		}
		Bprint(&bout, "\trm -f $(O%d)\n", k+1);
		Bprint(&bout, "\n");
	}

	// newpkg: clean
	//	6ar grc pkg.a
	Bprint(&bout, "newpkg: clean\n");
	for(i=0; i<npkg; i++)
		Bprint(&bout, "\t$(AR) grc %s.a\n", pkg[i]);
	Bprint(&bout, "\n");

	// $(O1): newpkg
	// $(O2): a1
	Bprint(&bout, "$(O1): newpkg\n");
	for(i=1; i<pass; i++)
		Bprint(&bout, "$(O%d): a%d\n", i+1, i);
	Bprint(&bout, "\n");

	// nuke: clean
	//	rm -f $(GOROOT)/pkg/xxx.a
	Bprint(&bout, "nuke: clean\n");
	Bprint(&bout, "\trm -f");
	for(i=0; i<npkg; i++)
		Bprint(&bout, " $(GOROOT)/pkg/%s.a", pkg[i]);
	Bprint(&bout, "\n\n");

	// packages: pkg.a
	//	rm -f $(GOROOT)/pkg/xxx.a
	Bprint(&bout, "packages:");
	for(i=0; i<npkg; i++)
		Bprint(&bout, " %s.a", pkg[i]);
	Bprint(&bout, "\n\n");

	// install: packages
	//	cp xxx.a $(GOROOT)/pkg/xxx.a
	Bprint(&bout, "install: packages\n");
	for(i=0; i<npkg; i++)
		Bprint(&bout, "\tcp %s.a $(GOROOT)/pkg/%s.a\n", pkg[i], pkg[i]);
	Bprint(&bout, "\n");

	Bterm(&bout);
}

int
sourcefilenames(char ***argvp)
{
	Dir *d;
	int dir, nd, i, argc;
	char **argv;

	if((dir = open(".", OREAD)) < 0)
		sysfatal("open .: %r");

	nd = dirreadall(dir, &d);
	close(dir);

	argv = emalloc((nd+1)*sizeof argv[0]);
	argc = 0;
	for(i=0; i<nd; i++) {
		if(suffix(d[i].name, ".go")
		|| suffix(d[i].name, ".c")
		|| suffix(d[i].name, ".s"))
			argv[argc++] = d[i].name;
	}
	*argvp = argv;
	argv[argc] = nil;
	return argc;
}

void
main(int argc, char **argv)
{
	int i, k, pass, npending, nfail, nsuccess, narfiles;
	Job **pending, **fail, **success, *j;
	char **arfiles;

	oargc = argc;
	oargv = argv;
	fmtinstall('$', dollarfmt);

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
	devnull = open("/dev/null", OWRITE);
	if(devnull < 0)
		sysfatal("open /dev/null: %r");
	goroot = getenv("GOROOT");
	if(goroot == nil)
		sysfatal("no $GOROOT");

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

	// If no arguments, use all source files in current directory.
	if(argc == 0)
		argc = sourcefilenames(&argv);

	// Make the job list.
	njob = 0;
	job = emalloc(argc*sizeof job[0]);
	for(i=0; i<argc; i++) {
		if(suffix(argv[i], "_test.go"))
			continue;
		job[njob].name = argv[i];
		job[njob].pass = -1;
		job[njob].pkg = getpkg(argv[i]);
		if(job[njob].pkg && strcmp(job[njob].pkg, "main") == 0)
			continue;
		njob++;
	}

	// Look for non-go files, which don't have packages.
	// If there's only one package in the go files, use it.
	for(i=0; i<njob; i++) {
		if(job[i].pkg == nil) {
			if(npkg == 1) {
				job[i].pkg = pkg[0];
				continue;
			}
			sysfatal("cannot determine package for %s", job[i].name);
		}
	}

	// TODO: subdirectory packages

	// Create empty archives for each package.
	for(i=0; i<npkg; i++) {
		unlink(smprint("%s.a", pkg[i]));
		ar(pkg[i], nil, 0);
	}

	// Compile by repeated passes: build as many .6 as you can,
	// put them in their archives, and repeat.
	pending = emalloc(njob*sizeof pending[0]);
	for(i=0; i<njob; i++)
		pending[i] = &job[i];
	npending = njob;

	fail = emalloc(njob*sizeof fail[0]);
	success = emalloc(njob*sizeof success[0]);
	arfiles = emalloc(njob*sizeof arfiles[0]);

	for(pass=0; npending > 0; pass++) {
		// Run what we can.
		nfail = 0;
		nsuccess = 0;
		for(i=0; i<npending; i++) {
			j = pending[i];
			if(buildcc(compiler(j->name), j->name, 0) < 0)
				fail[nfail++] = j;
			else{
				if(chatty == 1)
					fprint(2, "%s ", j->name);
				success[nsuccess++] = j;
			}
		}
		if(nsuccess == 0) {
			// Nothing ran; give up.
			for(i=0; i<nfail; i++) {
				j = fail[i];
				buildcc(compiler(j->name), j->name, 1);
			}
			exits("stalemate");
		}
		if(chatty == 1)
			fprint(2, "\n");

		// Update archives.
		for(i=0; i<npkg; i++) {
			narfiles = 0;
			for(k=0; k<nsuccess; k++) {
				j = success[k];
				if(j->pkg == pkg[i])
					arfiles[narfiles++] = goobj(j->name, thechar);
				j->pass = pass;
			}
			if(narfiles > 0)
				ar(pkg[i], arfiles, narfiles);
			for(k=0; k<narfiles; k++)
				unlink(arfiles[k]);
		}

		for(i=0; i<nfail; i++)
			pending[i] = fail[i];
		npending = nfail;
	}

	if(makefile)
		writemakefile();
	exits(0);
}
