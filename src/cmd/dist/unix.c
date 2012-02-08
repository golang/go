// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// These #ifdefs are being used as a substitute for
// build configuration, so that on any system, this
// tool can be built with the local equivalent of
//	cc *.c
//
#ifndef WIN32
#ifndef PLAN9

#include "a.h"
#include <unistd.h>
#include <dirent.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/param.h>
#include <sys/utsname.h>
#include <fcntl.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <stdarg.h>

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
			fatal("read: %s", strerror(errno));
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
	if(p != NULL)
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
} bg[MAXBG];
static int nbg;

static void bgwait1(void);

// genrun is the generic run implementation.
static void
genrun(Buf *b, char *dir, int mode, Vec *argv, int wait)
{
	int i, p[2], pid;
	Buf cmd;
	char *q;

	while(nbg >= nelem(bg))
		bgwait1();

	// Generate a copy of the command to show in a log.
	// Substitute $WORK for the work directory.
	binit(&cmd);
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
		xprintf("%s\n", bstr(&cmd));

	if(b != nil) {
		breset(b);
		if(pipe(p) < 0)
			fatal("pipe: %s", strerror(errno));
	}

	switch(pid = fork()) {
	case -1:
		fatal("fork: %s", strerror(errno));
	case 0:
		if(b != nil) {
			close(0);
			close(p[0]);
			dup2(p[1], 1);
			dup2(p[1], 2);
			if(p[1] > 2)
				close(p[1]);
		}
		if(dir != nil) {
			if(chdir(dir) < 0) {
				fprintf(stderr, "chdir %s: %s\n", dir, strerror(errno));
				_exit(1);
			}
		}
		vadd(argv, nil);
		execvp(argv->p[0], argv->p);
		fprintf(stderr, "%s\n", bstr(&cmd));
		fprintf(stderr, "exec %s: %s\n", argv->p[0], strerror(errno));
		_exit(1);
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
	nbg++;
	
	if(wait)
		bgwait();

	bfree(&cmd);
}

// bgwait1 waits for a single background job.
static void
bgwait1(void)
{
	int i, pid, status, mode;
	char *cmd;

	errno = 0;
	while((pid = wait(&status)) < 0) {
		if(errno != EINTR)
			fatal("waitpid: %s", strerror(errno));
	}
	for(i=0; i<nbg; i++)
		if(bg[i].pid == pid)
			goto ok;
	fatal("waitpid: unexpected pid");

ok:
	cmd = bg[i].cmd;
	mode = bg[i].mode;
	bg[i].pid = 0;
	bg[i] = bg[--nbg];
	
	if(mode == CheckExit && (!WIFEXITED(status) || WEXITSTATUS(status) != 0)) {
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
	char buf[MAXPATHLEN];
	
	breset(b);
	if(getcwd(buf, MAXPATHLEN) == nil)
		fatal("getcwd: %s", strerror(errno));
	bwritestr(b, buf);	
}

// xrealwd replaces b with the 'real' name for the given path.
// real is defined as what getcwd returns in that directory.
void
xrealwd(Buf *b, char *path)
{
	int fd;
	
	fd = open(".", 0);
	if(fd < 0)
		fatal("open .: %s", strerror(errno));
	if(chdir(path) < 0)
		fatal("chdir %s: %s", path, strerror(errno));
	xgetwd(b);
	if(fchdir(fd) < 0)
		fatal("fchdir: %s", strerror(errno));
	close(fd);
}

// isdir reports whether p names an existing directory.
bool
isdir(char *p)
{
	struct stat st;
	
	return stat(p, &st) >= 0 && S_ISDIR(st.st_mode);
}

// isfile reports whether p names an existing file.
bool
isfile(char *p)
{
	struct stat st;
	
	return stat(p, &st) >= 0 && S_ISREG(st.st_mode);
}

// mtime returns the modification time of the file p.
Time
mtime(char *p)
{
	struct stat st;
	
	if(stat(p, &st) < 0)
		return 0;
	return (Time)st.st_mtime*1000000000LL;
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
	fd = open(file, 0);
	if(fd < 0)
		fatal("open %s: %s", file, strerror(errno));
	breadfrom(b, fd);
	close(fd);
}

// writefile writes b to the named file, creating it if needed.
void
writefile(Buf *b, char *file)
{
	int fd;
	
	fd = creat(file, 0666);
	if(fd < 0)
		fatal("create %s: %s", file, strerror(errno));
	if(write(fd, b->p, b->len) != b->len)
		fatal("short write: %s", strerror(errno));
	close(fd);
}
	
// xmkdir creates the directory p.
void
xmkdir(char *p)
{
	if(mkdir(p, 0777) < 0)
		fatal("mkdir %s: %s", p, strerror(errno));
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
		xprintf("rm %s\n", p);
	unlink(p);
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
		if(vflag > 2)
			xprintf("rm %s\n", p);
		rmdir(p);
	} else {
		if(vflag > 2)
			xprintf("rm %s\n", p);
		unlink(p);
	}
	
	bfree(&b);
	vfree(&dir);	
}

// xreaddir replaces dst with a list of the names of the files in dir.
// The names are relative to dir; they are not full paths.
void
xreaddir(Vec *dst, char *dir)
{
	DIR *d;
	struct dirent *dp;
	
	vreset(dst);
	d = opendir(dir);
	if(d == nil)
		fatal("opendir %s: %s", dir, strerror(errno));
	while((dp = readdir(d)) != nil) {
		if(strcmp(dp->d_name, ".") == 0 || strcmp(dp->d_name, "..") == 0)
			continue;
		vadd(dst, dp->d_name);
	}
	closedir(d);
}

// xworkdir creates a new temporary directory to hold object files
// and returns the name of that directory.
char*
xworkdir(void)
{
	Buf b;
	char *p;
	
	binit(&b);

	xgetenv(&b, "TMPDIR");
	if(b.len == 0)
		bwritestr(&b, "/var/tmp");
	bwritestr(&b, "/go-cbuild-XXXXXX");
	if(mkdtemp(bstr(&b)) == nil)
		fatal("mkdtemp: %s", strerror(errno));
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
	exit(1);
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
	return np >= ns && strcmp(p+np-ns, suffix) == 0;
}

// hasprefix reports whether p begins wtih prefix.
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
	exit(n);
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

// xsetenv sets the environment variable $name to the given value.
void
xsetenv(char *name, char *value)
{
	setenv(name, value, 1);
}

// main takes care of OS-specific startup and dispatches to xmain.
int
main(int argc, char **argv)
{
	Buf b;
	struct utsname u;

	binit(&b);
	
	slash = "/";

#if defined(__APPLE__)
	gohostos = "darwin";
	// Even on 64-bit platform, darwin uname -m prints i386.
	run(&b, nil, 0, "sysctl", "machdep.cpu.extfeatures", nil);
	if(contains(bstr(&b), "EM64T"))
		gohostarch = "amd64";
#elif defined(__linux__)
	gohostos = "linux";
#elif defined(__FreeBSD__)
	gohostos = "freebsd";
#elif defined(__OpenBSD__)
	gohostos = "openbsd";
#elif defined(__NetBSD__)
	gohostos = "netbsd";
#else
	fatal("unknown operating system");
#endif

	if(gohostarch == nil) {
		if(uname(&u) < 0)
			fatal("uname: %s", strerror(errno));
		if(contains(u.machine, "x86_64") || contains(u.machine, "amd64"))
			gohostarch = "amd64";
		else if(hassuffix(u.machine, "86"))
			gohostarch = "386";
		else if(contains(u.machine, "arm"))
			gohostarch = "arm";
		else
			fatal("unknown architecture: %s", u.machine);
	}

	init();
	xmain(argc, argv);
	bfree(&b);
	return 0;
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

#endif // PLAN9
#endif // __WINDOWS__
