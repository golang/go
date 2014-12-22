// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// These #ifdefs are being used as a substitute for
// build configuration, so that on any system, this
// tool can be built with the local equivalent of
//	cc *.c
//
#ifdef PLAN9

#include <u.h>
#include <libc.h>
#include <stdio.h>
#undef nil
#undef nelem
#include "a.h"

// bprintf replaces the buffer with the result of the printf formatting
// and returns a pointer to the NUL-terminated buffer contents.
char*
bprintf(Buf *b, char *fmt, ...)
{
	va_list arg;
	char buf[4096];

	breset(b);
	va_start(arg, fmt);
	vsnprintf(buf, sizeof buf, fmt, arg);
	va_end(arg);
	bwritestr(b, buf);
	return bstr(b);
}

// bpathf is the same as bprintf (on windows it turns / into \ after the printf).
// It returns a pointer to the NUL-terminated buffer contents.
char*
bpathf(Buf *b, char *fmt, ...)
{
	va_list arg;
	char buf[4096];
	
	breset(b);
	va_start(arg, fmt);
	vsnprintf(buf, sizeof buf, fmt, arg);
	va_end(arg);
	bwritestr(b, buf);
	return bstr(b);
}

// bwritef is like bprintf but does not reset the buffer
// and does not return the NUL-terminated string.
void
bwritef(Buf *b, char *fmt, ...)
{
	va_list arg;
	char buf[4096];
	
	va_start(arg, fmt);
	vsnprintf(buf, sizeof buf, fmt, arg);
	va_end(arg);
	bwritestr(b, buf);
}

// breadfrom appends to b all the data that can be read from fd.
static void
breadfrom(Buf *b, int fd)
{
	int n;

	for(;;) {
		bgrow(b, 4096);
		n = read(fd, b->p+b->len, 4096);
		if(n < 0)
			fatal("read");
		if(n == 0)
			break;
		b->len += n;
	}
}

// xgetenv replaces b with the value of the named environment variable.
void
xgetenv(Buf *b, char *name)
{
	char *p;
	
	breset(b);
	p = getenv(name);
	if(p != nil)
		bwritestr(b, p);
}

static void genrun(Buf *b, char *dir, int mode, Vec *argv, int bg);

// run runs the command named by cmd.
// If b is not nil, run replaces b with the output of the command.
// If dir is not nil, run runs the command in that directory.
// If mode is CheckExit, run calls fatal if the command is not successful.
void
run(Buf *b, char *dir, int mode, char *cmd, ...)
{
	va_list arg;
	Vec argv;
	char *p;
	
	vinit(&argv);
	vadd(&argv, cmd);
	va_start(arg, cmd);
	while((p = va_arg(arg, char*)) != nil)
		vadd(&argv, p);
	va_end(arg);
	
	runv(b, dir, mode, &argv);
	
	vfree(&argv);
}

// runv is like run but takes a vector.
void
runv(Buf *b, char *dir, int mode, Vec *argv)
{
	genrun(b, dir, mode, argv, 1);
}

// bgrunv is like run but runs the command in the background.
// bgwait waits for pending bgrunv to finish.
void
bgrunv(char *dir, int mode, Vec *argv)
{
	genrun(nil, dir, mode, argv, 0);
}

#define MAXBG 4 /* maximum number of jobs to run at once */

static struct {
	int pid;
	int mode;
	char *cmd;
	Buf *b;
} bg[MAXBG];
static int nbg;
static int maxnbg = nelem(bg);

static void bgwait1(void);

// genrun is the generic run implementation.
static void
genrun(Buf *b, char *dir, int mode, Vec *argv, int wait)
{
	int i, p[2], pid;
	Buf b1, cmd;
	char *q;

	while(nbg >= maxnbg)
		bgwait1();

	binit(&b1);
	binit(&cmd);

	if(!isabs(argv->p[0])) {
		bpathf(&b1, "/bin/%s", argv->p[0]);
		free(argv->p[0]);
		argv->p[0] = xstrdup(bstr(&b1));
	}

	// Generate a copy of the command to show in a log.
	// Substitute $WORK for the work directory.
	for(i=0; i<argv->len; i++) {
		if(i > 0)
			bwritestr(&cmd, " ");
		q = argv->p[i];
		if(workdir != nil && hasprefix(q, workdir)) {
			bwritestr(&cmd, "$WORK");
			q += strlen(workdir);
		}
		bwritestr(&cmd, q);
	}
	if(vflag > 1)
		errprintf("%s\n", bstr(&cmd));

	if(b != nil) {
		breset(b);
		if(pipe(p) < 0)
			fatal("pipe");
	}

	switch(pid = fork()) {
	case -1:
		fatal("fork");
	case 0:
		if(b != nil) {
			close(0);
			close(p[0]);
			dup(p[1], 1);
			dup(p[1], 2);
			if(p[1] > 2)
				close(p[1]);
		}
		if(dir != nil) {
			if(chdir(dir) < 0) {
				fprint(2, "chdir: %r\n");
				_exits("chdir");
			}
		}
		vadd(argv, nil);
		exec(argv->p[0], argv->p);
		fprint(2, "%s\n", bstr(&cmd));
		fprint(2, "exec: %r\n");
		_exits("exec");
	}
	if(b != nil) {
		close(p[1]);
		breadfrom(b, p[0]);
		close(p[0]);
	}

	if(nbg < 0)
		fatal("bad bookkeeping");
	bg[nbg].pid = pid;
	bg[nbg].mode = mode;
	bg[nbg].cmd = btake(&cmd);
	bg[nbg].b = b;
	nbg++;
	
	if(wait)
		bgwait();

	bfree(&cmd);
	bfree(&b1);
}

// bgwait1 waits for a single background job.
static void
bgwait1(void)
{
	Waitmsg *w;
	int i, mode;
	char *cmd;
	Buf *b;

	w = wait();
	if(w == nil)
		fatal("wait");
		
	for(i=0; i<nbg; i++)
		if(bg[i].pid == w->pid)
			goto ok;
	fatal("wait: unexpected pid");

ok:
	cmd = bg[i].cmd;
	mode = bg[i].mode;
	bg[i].pid = 0;
	b = bg[i].b;
	bg[i].b = nil;
	bg[i] = bg[--nbg];
	
	if(mode == CheckExit && w->msg[0]) {
		if(b != nil)
			xprintf("%s\n", bstr(b));
		fatal("FAILED: %s", cmd);
	}
	xfree(cmd);
}

// bgwait waits for all the background jobs.
void
bgwait(void)
{
	while(nbg > 0)
		bgwait1();
}

// xgetwd replaces b with the current directory.
void
xgetwd(Buf *b)
{
	char buf[4096];
	
	breset(b);
	if(getwd(buf, sizeof buf) == nil)
		fatal("getwd");
	bwritestr(b, buf);
}

// xrealwd replaces b with the 'real' name for the given path.
// real is defined as what getcwd returns in that directory.
void
xrealwd(Buf *b, char *path)
{
	char buf[4096];
	int fd;

	fd = open(path, OREAD);
	if(fd2path(fd, buf, sizeof buf) < 0)
		fatal("fd2path");
	close(fd);
	breset(b);
	bwritestr(b, buf);
}

// isdir reports whether p names an existing directory.
bool
isdir(char *p)
{
	Dir *d;
	ulong mode;

	d = dirstat(p);
	if(d == nil)
		return 0;
	mode = d->mode;
	free(d);
	return (mode & DMDIR) == DMDIR;
}

// isfile reports whether p names an existing file.
bool
isfile(char *p)
{
	Dir *d;
	ulong mode;

	d = dirstat(p);
	if(d == nil)
		return 0;
	mode = d->mode;
	free(d);
	return (mode & DMDIR) == 0;
}

// mtime returns the modification time of the file p.
Time
mtime(char *p)
{
	Dir *d;
	ulong t;

	d = dirstat(p);
	if(d == nil)
		return 0;
	t = d->mtime;
	free(d);
	return (Time)t;
}

// isabs reports whether p is an absolute path.
bool
isabs(char *p)
{
	return hasprefix(p, "/");
}

// readfile replaces b with the content of the named file.
void
readfile(Buf *b, char *file)
{
	int fd;

	breset(b);
	fd = open(file, OREAD);
	if(fd < 0)
		fatal("open %s", file);
	breadfrom(b, fd);
	close(fd);
}

// writefile writes b to the named file, creating it if needed.
void
writefile(Buf *b, char *file, int exec)
{
	int fd;
	Dir d;
	
	fd = create(file, ORDWR, 0666);
	if(fd < 0)
		fatal("create %s", file);
	if(write(fd, b->p, b->len) != b->len)
		fatal("short write");
	if(exec) {
		nulldir(&d);
		d.mode = 0755;
		dirfwstat(fd, &d);
	}
	close(fd);
}

// xmkdir creates the directory p.
void
xmkdir(char *p)
{
	int fd;

	if(isdir(p))
		return;
	fd = create(p, OREAD, 0777|DMDIR);
	close(fd);
	if(fd < 0)
		fatal("mkdir %s", p);
}

// xmkdirall creates the directory p and its parents, as needed.
void
xmkdirall(char *p)
{
	char *q;

	if(isdir(p))
		return;
	q = strrchr(p, '/');
	if(q != nil) {
		*q = '\0';
		xmkdirall(p);
		*q = '/';
	}
	xmkdir(p);
}

// xremove removes the file p.
void
xremove(char *p)
{
	if(vflag > 2)
		errprintf("rm %s\n", p);
	remove(p);
}

// xremoveall removes the file or directory tree rooted at p.
void
xremoveall(char *p)
{
	int i;
	Buf b;
	Vec dir;

	binit(&b);
	vinit(&dir);

	if(isdir(p)) {
		xreaddir(&dir, p);
		for(i=0; i<dir.len; i++) {
			bprintf(&b, "%s/%s", p, dir.p[i]);
			xremoveall(bstr(&b));
		}
	}
	if(vflag > 2)
		errprintf("rm %s\n", p);
	remove(p);
	
	bfree(&b);
	vfree(&dir);	
}

// xreaddir replaces dst with a list of the names of the files in dir.
// The names are relative to dir; they are not full paths.
void
xreaddir(Vec *dst, char *dir)
{
	Dir *d;
	int fd, i, n;

	vreset(dst);

	fd = open(dir, OREAD);
	if(fd < 0)
		fatal("open %s", dir);
	n = dirreadall(fd, &d);
	for(i=0; i<n; i++)
		vadd(dst, d[i].name);
	free(d);
	close(fd);
}

// xworkdir creates a new temporary directory to hold object files
// and returns the name of that directory.
char*
xworkdir(void)
{
	Buf b;
	char *p;
	int fd, tries;

	binit(&b);

	fd = 0;
	for(tries=0; tries<1000; tries++) {
		bprintf(&b, "/tmp/go-cbuild-%06x", nrand((1<<24)-1));
		fd = create(bstr(&b), OREAD|OEXCL, 0700|DMDIR);
		if(fd >= 0)
			goto done;
	}
	fatal("xworkdir create");

done:
	close(fd);
	p = btake(&b);

	bfree(&b);
	return p;
}

// fatal prints an error message to standard error and exits.
void
fatal(char *msg, ...)
{
	va_list arg;
	
	fflush(stdout);
	fprintf(stderr, "go tool dist: ");
	va_start(arg, msg);
	vfprintf(stderr, msg, arg);
	va_end(arg);
	fprintf(stderr, "\n");

	bgwait();
	exits(msg);
}

// xmalloc returns a newly allocated zeroed block of n bytes of memory.
// It calls fatal if it runs out of memory.
void*
xmalloc(int n)
{
	void *p;
	
	p = malloc(n);
	if(p == nil)
		fatal("out of memory");
	memset(p, 0, n);
	return p;
}

// xstrdup returns a newly allocated copy of p.
// It calls fatal if it runs out of memory.
char*
xstrdup(char *p)
{
	p = strdup(p);
	if(p == nil)
		fatal("out of memory");
	return p;
}

// xrealloc grows the allocation p to n bytes and
// returns the new (possibly moved) pointer.
// It calls fatal if it runs out of memory.
void*
xrealloc(void *p, int n)
{
	p = realloc(p, n);
	if(p == nil)
		fatal("out of memory");
	return p;
}

// xfree frees the result returned by xmalloc, xstrdup, or xrealloc.
void
xfree(void *p)
{
	free(p);
}

// hassuffix reports whether p ends with suffix.
bool
hassuffix(char *p, char *suffix)
{
	int np, ns;

	np = strlen(p);
	ns = strlen(suffix);
	return np >= ns && streq(p+np-ns, suffix);
}

// hasprefix reports whether p begins with prefix.
bool
hasprefix(char *p, char *prefix)
{
	return strncmp(p, prefix, strlen(prefix)) == 0;
}

// contains reports whether sep appears in p.
bool
contains(char *p, char *sep)
{
	return strstr(p, sep) != nil;
}

// streq reports whether p and q are the same string.
bool
streq(char *p, char *q)
{
	return strcmp(p, q) == 0;
}

// lastelem returns the final path element in p.
char*
lastelem(char *p)
{
	char *out;

	out = p;
	for(; *p; p++)
		if(*p == '/')
			out = p+1;
	return out;
}

// xmemmove copies n bytes from src to dst.
void
xmemmove(void *dst, void *src, int n)
{
	memmove(dst, src, n);
}

// xmemcmp compares the n-byte regions starting at a and at b.
int
xmemcmp(void *a, void *b, int n)
{
	return memcmp(a, b, n);
}

// xstrlen returns the length of the NUL-terminated string at p.
int
xstrlen(char *p)
{
	return strlen(p);
}

// xexit exits the process with return code n.
void
xexit(int n)
{
	char buf[32];

	snprintf(buf, sizeof buf, "%d", n);
	exits(buf);
}

// xatexit schedules the exit-handler f to be run when the program exits.
void
xatexit(void (*f)(void))
{
	atexit(f);
}

// xprintf prints a message to standard output.
void
xprintf(char *fmt, ...)
{
	va_list arg;
	
	va_start(arg, fmt);
	vprintf(fmt, arg);
	va_end(arg);
}

// errprintf prints a message to standard output.
void
errprintf(char *fmt, ...)
{
	va_list arg;
	
	va_start(arg, fmt);
	vfprintf(stderr, fmt, arg);
	va_end(arg);
}

// xsetenv sets the environment variable $name to the given value.
void
xsetenv(char *name, char *value)
{
	putenv(name, value);
}

// main takes care of OS-specific startup and dispatches to xmain.
void
main(int argc, char **argv)
{
	Buf b;

	setvbuf(stdout, nil, _IOLBF, BUFSIZ);
	setvbuf(stderr, nil, _IOLBF, BUFSIZ);

	binit(&b);

	rfork(RFENVG);

	slash = "/";
	gohostos = "plan9";

	xgetenv(&b, "objtype");
	if(b.len == 0)
		fatal("$objtype is unset");
	gohostarch = btake(&b);

	srand(time(0)+getpid());
	init();
	xmain(argc, argv);

	bfree(&b);
	exits(nil);
}

// xqsort is a wrapper for the C standard qsort.
void
xqsort(void *data, int n, int elemsize, int (*cmp)(const void*, const void*))
{
	qsort(data, n, elemsize, cmp);
}

// xstrcmp compares the NUL-terminated strings a and b.
int
xstrcmp(char *a, char *b)
{
	return strcmp(a, b);
}

// xstrstr returns a pointer to the first occurrence of b in a.
char*
xstrstr(char *a, char *b)
{
	return strstr(a, b);
}

// xstrrchr returns a pointer to the final occurrence of c in p.
char*
xstrrchr(char *p, int c)
{
	return strrchr(p, c);
}

// xsamefile reports whether f1 and f2 are the same file (or dir)
int
xsamefile(char *f1, char *f2)
{
	return streq(f1, f2); // suffice for now
}

// xtryexecfunc tries to execute function f, if any illegal instruction
// signal received in the course of executing that function, it will
// return 0, otherwise it will return 1.
int
xtryexecfunc(void (*f)(void))
{
	USED(f);
	return 0; // suffice for now
}

bool
cansse2(void)
{
	// if we had access to cpuid, could answer this question
	// less conservatively.
	return 0;
}

#endif // PLAN9
