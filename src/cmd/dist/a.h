// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

typedef int bool;

// The Time unit is unspecified; we just need to
// be able to compare whether t1 is older than t2 with t1 < t2.
typedef long long Time;

#define nil ((void*)0)
#define nelem(x) (sizeof(x)/sizeof((x)[0]))
#ifndef PLAN9
#define USED(x) ((void)(x))
#endif

// A Buf is a byte buffer, like Go's []byte.
typedef struct Buf Buf;
struct Buf
{
	char *p;
	int len;
	int cap;
};

// A Vec is a string vector, like Go's []string.
typedef struct Vec Vec;
struct Vec
{
	char **p;
	int len;
	int cap;
};

// Modes for run.
enum {
	CheckExit = 1,
};

// buf.c
bool	bequal(Buf *s, Buf *t);
void	bsubst(Buf *b, char *x, char *y);
void	bfree(Buf *b);
void	bgrow(Buf *b, int n);
void	binit(Buf *b);
char*	bpathf(Buf *b, char *fmt, ...);
char*	bprintf(Buf *b, char *fmt, ...);
void	bwritef(Buf *b, char *fmt, ...);
void	breset(Buf *b);
char*	bstr(Buf *b);
char*	btake(Buf *b);
void	bwrite(Buf *b, void *v, int n);
void	bwriteb(Buf *dst, Buf *src);
void	bwritestr(Buf *b, char *p);
void	bswap(Buf *b, Buf *b1);
void	vadd(Vec *v, char *p);
void	vcopy(Vec *dst, char **src, int n);
void	vfree(Vec *v);
void	vgrow(Vec *v, int n);
void	vinit(Vec *v);
void	vreset(Vec *v);
void	vuniq(Vec *v);
void	splitlines(Vec*, char*);
void	splitfields(Vec*, char*);

// build.c
extern char *goarch;
extern char *gobin;
extern char *gochar;
extern char *gohostarch;
extern char *gohostos;
extern char *goos;
extern char *goroot;
extern char *goroot_final;
extern char *goextlinkenabled;
extern char *goversion;
extern char *defaultcc;
extern char *defaultcxxtarget;
extern char *defaultcctarget;
extern char *workdir;
extern char *tooldir;
extern char *slash;
extern bool rebuildall;
extern bool defaultclang;

int	find(char*, char**, int);
void	init(void);
void	cmdbanner(int, char**);
void	cmdbootstrap(int, char**);
void	cmdclean(int, char**);
void	cmdenv(int, char**);
void	cmdinstall(int, char**);
void	cmdversion(int, char**);

// buildgc.c
void	gcopnames(char*, char*);
void	mkanames(char*, char*);

// buildruntime.c
void	mkzasm(char*, char*);
void	mkzsys(char*, char*);
void	mkzgoarch(char*, char*);
void	mkzgoos(char*, char*);
void	mkzruntimedefs(char*, char*);
void	mkzversion(char*, char*);
void	mkzexperiment(char*, char*);

// buildgo.c
void	mkzdefaultcc(char*, char*);

// main.c
extern int vflag;
extern int sflag;
void	usage(void);
void	xmain(int argc, char **argv);

// portability layer (plan9.c, unix.c, windows.c)
bool	contains(char *p, char *sep);
void	errprintf(char*, ...);
void	fatal(char *msg, ...);
bool	hasprefix(char *p, char *prefix);
bool	hassuffix(char *p, char *suffix);
bool	isabs(char*);
bool	isdir(char *p);
bool	isfile(char *p);
char*	lastelem(char*);
Time	mtime(char*);
void	readfile(Buf*, char*);
void	copyfile(char*, char*, int);
void	run(Buf *b, char *dir, int mode, char *cmd, ...);
void	runv(Buf *b, char *dir, int mode, Vec *argv);
void	bgrunv(char *dir, int mode, Vec *argv);
void	bgwait(void);
bool	streq(char*, char*);
bool	cansse2(void);
void	writefile(Buf*, char*, int);
void	xatexit(void (*f)(void));
void	xexit(int);
void	xfree(void*);
void	xgetenv(Buf *b, char *name);
void	xgetwd(Buf *b);
void*	xmalloc(int n);
void*	xmalloc(int);
int	xmemcmp(void*, void*, int);
void	xmemmove(void*, void*, int);
void	xmkdir(char *p);
void	xmkdirall(char*);
Time	xmtime(char *p);
void	xprintf(char*, ...);
void	xqsort(void*, int, int, int(*)(const void*, const void*));
void	xreaddir(Vec *dst, char *dir);
void*	xrealloc(void*, int);
void	xrealwd(Buf *b, char *path);
void	xremove(char *p);
void	xremoveall(char *p);
void	xsetenv(char*, char*);
int	xstrcmp(char*, char*);
char*	xstrdup(char *p);
int	xstrlen(char*);
char*	xstrrchr(char*, int);
char*	xstrstr(char*, char*);
char*	xworkdir(void);
int	xsamefile(char*, char*);
char*	xgetgoarm(void);
int	xtryexecfunc(void (*)(void));
