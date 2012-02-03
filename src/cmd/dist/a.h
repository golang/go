// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

typedef int bool;

// The Time unit is unspecified; we just need to
// be able to compare whether t1 is older than t2 with t1 < t2.
typedef long long Time;

#define nil ((void*)0)
#define nelem(x) (sizeof(x)/sizeof((x)[0]))
#define USED(x) ((void)(x))

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
void	bfree(Buf *b);
void	bgrow(Buf *b, int n);
void	binit(Buf *b);
char*	bprintf(Buf *b, char *fmt, ...);
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
extern char *default_goroot;
extern char *goarch;
extern char *gobin;
extern char *gohostarch;
extern char *gohostos;
extern char *goos;
extern char *goroot;
extern char *workdir;
extern char *slash;

void	init(void);
void	cmdbootstrap(int, char**);
void	cmdenv(int, char**);
void	cmdinstall(int, char**);

// buildgc.c
void	gcopnames(char*, char*);
void	mkenam(char*, char*);

// main.c
void	xmain(int argc, char **argv);

// portability layer (plan9.c, unix.c, windows.c)
bool	contains(char *p, char *sep);
void	fatal(char *msg, ...);
bool	hasprefix(char *p, char *prefix);
bool	hassuffix(char *p, char *suffix);
bool	isabs(char*);
bool	isdir(char *p);
bool	isfile(char *p);
char*	lastelem(char*);
Time	mtime(char*);
void	readfile(Buf*, char*);
void	run(Buf *b, char *dir, int mode, char *cmd, ...);
void	runv(Buf *b, char *dir, int mode, Vec *argv);
bool	streq(char*, char*);
void	writefile(Buf*, char*);
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
int	xstreq(char*, char*);
int	xstrlen(char*);
char*	xstrrchr(char*, int);
char*	xstrstr(char*, char*);
char*	xworkdir(void);
