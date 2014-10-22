// Copyright 2012 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// These #ifdefs are being used as a substitute for
// build configuration, so that on any system, this
// tool can be built with the local equivalent of
//	cc *.c
//
#ifdef WIN32

// Portability layer implemented for Windows.
// See unix.c for doc comments about exported functions.

#include "a.h"
#include <windows.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

/*
 * Windows uses 16-bit rune strings in the APIs.
 * Define conversions between Rune* and UTF-8 char*.
 */

typedef unsigned char uchar;
typedef unsigned short Rune;  // same as Windows

// encoderune encodes the rune r into buf and returns
// the number of bytes used.
static int
encoderune(char *buf, Rune r)
{
	if(r < 0x80) {  // 7 bits
		buf[0] = r;
		return 1;
	}
	if(r < 0x800) {  // 5+6 bits
		buf[0] = 0xc0 | (r>>6);
		buf[1] = 0x80 | (r&0x3f);
		return 2;
	}
	buf[0] = 0xe0 | (r>>12);
	buf[1] = 0x80 | ((r>>6)&0x3f);
	buf[2] = 0x80 | (r&0x3f);
	return 3;
}

// decoderune decodes the rune encoding at sbuf into r
// and returns the number of bytes used.
static int
decoderune(Rune *r, char *sbuf)
{
	uchar *buf;

	buf = (uchar*)sbuf;
	if(buf[0] < 0x80) {
		*r = buf[0];
		return 1;
	}
	if((buf[0]&0xe0) == 0xc0 && (buf[1]&0xc0) == 0x80) {
		*r = (buf[0]&~0xc0)<<6 | (buf[1]&~0x80);
		if(*r < 0x80)
			goto err;
		return 2;
	}
	if((buf[0]&0xf0) == 0xe0 && (buf[1]&0xc0) == 0x80 && (buf[2]&0xc0) == 0x80) {
		*r = (buf[0]&~0xc0)<<12 | (buf[1]&~0x80)<<6 | (buf[2]&~0x80);
		if(*r < 0x800)
			goto err;
		return 3;
	}
err:
	*r = 0xfffd;
	return 1;
}

// toutf replaces b with the UTF-8 encoding of the rune string r.	
static void
toutf(Buf *b, Rune *r)
{
	int i, n;
	char buf[4];

	breset(b);
	for(i=0; r[i]; i++) {
		n = encoderune(buf, r[i]);
		bwrite(b, buf, n);
	}
}

// torune replaces *rp with a pointer to a newly allocated
// rune string equivalent of the UTF-8 string p.
static void
torune(Rune **rp, char *p)
{
	Rune *r, *w;

	r = xmalloc((strlen(p)+1) * sizeof r[0]);
	w = r;
	while(*p)
		p += decoderune(w++, p);
	*w = 0;
	*rp = r;
}

// errstr returns the most recent Windows error, in string form.
static char*
errstr(void)
{
	DWORD code;
	Rune *r;
	Buf b;

	binit(&b);
	code = GetLastError();
	r = nil;
	FormatMessageW(FORMAT_MESSAGE_ALLOCATE_BUFFER|FORMAT_MESSAGE_FROM_SYSTEM|FORMAT_MESSAGE_IGNORE_INSERTS,
		nil, code, 0, (Rune*)&r, 0, nil);
	toutf(&b, r);
	return bstr(&b);  // leak but we're dying anyway
}

void
xgetenv(Buf *b, char *name)
{
	Rune *buf;
	int n;
	Rune *r;

	breset(b);
	torune(&r, name);
	n = GetEnvironmentVariableW(r, NULL, 0);
	if(n > 0) {
		buf = xmalloc((n+1)*sizeof buf[0]);
		GetEnvironmentVariableW(r, buf, n+1);
		buf[n] = '\0';
		toutf(b, buf);
		xfree(buf);
	}
	xfree(r);
}

void
xsetenv(char *name, char *value)
{
	Rune *rname, *rvalue;

	torune(&rname, name);
	torune(&rvalue, value);
	SetEnvironmentVariableW(rname, rvalue);
	xfree(rname);
	xfree(rvalue);
}

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

void
bwritef(Buf *b, char *fmt, ...)
{
	va_list arg;
	char buf[4096];
	
	// no reset
	va_start(arg, fmt);
	vsnprintf(buf, sizeof buf, fmt, arg);
	va_end(arg);
	bwritestr(b, buf);
}

// bpathf is like bprintf but replaces / with \ in the result,
// to make it a canonical windows file path.
char*
bpathf(Buf *b, char *fmt, ...)
{
	int i;
	va_list arg;
	char buf[4096];
	
	breset(b);
	va_start(arg, fmt);
	vsnprintf(buf, sizeof buf, fmt, arg);
	va_end(arg);
	bwritestr(b, buf);

	for(i=0; i<b->len; i++)
		if(b->p[i] == '/')
			b->p[i] = '\\';

	return bstr(b);
}


static void
breadfrom(Buf *b, HANDLE h)
{
	DWORD n;

	for(;;) {
		if(b->len > 1<<22)
			fatal("unlikely file size in readfrom");
		bgrow(b, 4096);
		n = 0;
		if(!ReadFile(h, b->p+b->len, 4096, &n, nil)) {
			// Happens for pipe reads.
			break;
		}
		if(n == 0)
			break;
		b->len += n;
	}
}

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

static void genrun(Buf*, char*, int, Vec*, int);

void
runv(Buf *b, char *dir, int mode, Vec *argv)
{
	genrun(b, dir, mode, argv, 1);
}

void
bgrunv(char *dir, int mode, Vec *argv)
{
	genrun(nil, dir, mode, argv, 0);
}

#define MAXBG 4 /* maximum number of jobs to run at once */

static struct {
	PROCESS_INFORMATION pi;
	int mode;
	char *cmd;
} bg[MAXBG];

static int nbg;

static void bgwait1(void);

static void
genrun(Buf *b, char *dir, int mode, Vec *argv, int wait)
{
	// Another copy of this logic is in ../../lib9/run_windows.c.
	// If there's a bug here, fix the logic there too.
	int i, j, nslash;
	Buf cmd;
	char *q;
	Rune *rcmd, *rexe, *rdir;
	STARTUPINFOW si;
	PROCESS_INFORMATION pi;
	HANDLE p[2];

	while(nbg >= nelem(bg))
		bgwait1();

	binit(&cmd);

	for(i=0; i<argv->len; i++) {
		q = argv->p[i];
		if(i == 0 && streq(q, "hg"))
			bwritestr(&cmd, "cmd.exe /c ");
		if(i > 0)
			bwritestr(&cmd, " ");
		if(contains(q, " ") || contains(q, "\t") || contains(q, "\"") || contains(q, "\\\\") || hassuffix(q, "\\")) {
			bwritestr(&cmd, "\"");
			nslash = 0;
			for(; *q; q++) {
				if(*q == '\\') {
					nslash++;
					continue;
				}
				if(*q == '"') {
					for(j=0; j<2*nslash+1; j++)
						bwritestr(&cmd, "\\");
					nslash = 0;
				}
				for(j=0; j<nslash; j++)
					bwritestr(&cmd, "\\");
				nslash = 0;
				bwrite(&cmd, q, 1);
			}
			for(j=0; j<2*nslash; j++)
				bwritestr(&cmd, "\\");
			bwritestr(&cmd, "\"");
		} else {
			bwritestr(&cmd, q);
		}
	}
	if(vflag > 1)
		errprintf("%s\n", bstr(&cmd));

	torune(&rcmd, bstr(&cmd));
	rexe = nil;
	rdir = nil;
	if(dir != nil)
		torune(&rdir, dir);

	memset(&si, 0, sizeof si);
	si.cb = sizeof si;
	si.dwFlags = STARTF_USESTDHANDLES;
	si.hStdInput = INVALID_HANDLE_VALUE;
	if(b == nil) {
		si.hStdOutput = GetStdHandle(STD_OUTPUT_HANDLE);
		si.hStdError = GetStdHandle(STD_ERROR_HANDLE);
	} else {
		SECURITY_ATTRIBUTES seci;

		memset(&seci, 0, sizeof seci);
		seci.nLength = sizeof seci;
		seci.bInheritHandle = 1;
		breset(b);
		if(!CreatePipe(&p[0], &p[1], &seci, 0))
			fatal("CreatePipe: %s", errstr());
		si.hStdOutput = p[1];
		si.hStdError = p[1];
	}

	if(!CreateProcessW(rexe, rcmd, nil, nil, TRUE, 0, nil, rdir, &si, &pi)) {
		if(mode!=CheckExit)
			return;
		fatal("%s: %s", argv->p[0], errstr());
	}
	if(rexe != nil)
		xfree(rexe);
	xfree(rcmd);
	if(rdir != nil)
		xfree(rdir);
	if(b != nil) {
		CloseHandle(p[1]);
		breadfrom(b, p[0]);
		CloseHandle(p[0]);
	}

	if(nbg < 0)
		fatal("bad bookkeeping");
	bg[nbg].pi = pi;
	bg[nbg].mode = mode;
	bg[nbg].cmd = btake(&cmd);
	nbg++;

	if(wait)
		bgwait();

	bfree(&cmd);
}

// closes the background job for bgwait1
static void
bgwaitclose(int i)
{
	if(i < 0 || i >= nbg)
		return;

	CloseHandle(bg[i].pi.hProcess);
	CloseHandle(bg[i].pi.hThread);
	
	bg[i] = bg[--nbg];
}

// bgwait1 waits for a single background job
static void
bgwait1(void)
{
	int i, mode;
	char *cmd;
	HANDLE bgh[MAXBG];
	DWORD code;

	if(nbg == 0)
		fatal("bgwait1: nothing left");

	for(i=0; i<nbg; i++)
		bgh[i] = bg[i].pi.hProcess;
	i = WaitForMultipleObjects(nbg, bgh, FALSE, INFINITE);
	if(i < 0 || i >= nbg)
		fatal("WaitForMultipleObjects: %s", errstr());

	cmd = bg[i].cmd;
	mode = bg[i].mode;
	if(!GetExitCodeProcess(bg[i].pi.hProcess, &code)) {
		bgwaitclose(i);
		fatal("GetExitCodeProcess: %s", errstr());
		return;
	}

	if(mode==CheckExit && code != 0) {
		bgwaitclose(i);
		fatal("FAILED: %s", cmd);
		return;
	}

	bgwaitclose(i);
}

void
bgwait(void)
{
	while(nbg > 0)
		bgwait1();
}

// rgetwd returns a rune string form of the current directory's path.
static Rune*
rgetwd(void)
{
	int n;
	Rune *r;

	n = GetCurrentDirectoryW(0, nil);
	r = xmalloc((n+1)*sizeof r[0]);
	GetCurrentDirectoryW(n+1, r);
	r[n] = '\0';
	return r;
}

void
xgetwd(Buf *b)
{
	Rune *r;

	r = rgetwd();
	breset(b);
	toutf(b, r);
	xfree(r);
}

void
xrealwd(Buf *b, char *path)
{
	Rune *old;
	Rune *rnew;

	old = rgetwd();
	torune(&rnew, path);
	if(!SetCurrentDirectoryW(rnew))
		fatal("chdir %s: %s", path, errstr());
	xfree(rnew);
	xgetwd(b);
	if(!SetCurrentDirectoryW(old)) {
		breset(b);
		toutf(b, old);
		fatal("chdir %s: %s", bstr(b), errstr());
	}
}

bool
isdir(char *p)
{
	DWORD attr;
	Rune *r;

	torune(&r, p);
	attr = GetFileAttributesW(r);
	xfree(r);
	return attr != INVALID_FILE_ATTRIBUTES && (attr & FILE_ATTRIBUTE_DIRECTORY);
}

bool
isfile(char *p)
{
	DWORD attr;
	Rune *r;

	torune(&r, p);
	attr = GetFileAttributesW(r);
	xfree(r);
	return attr != INVALID_FILE_ATTRIBUTES && !(attr & FILE_ATTRIBUTE_DIRECTORY);
}

Time
mtime(char *p)
{
	HANDLE h;
	WIN32_FIND_DATAW data;
	Rune *r;
	FILETIME *ft;

	torune(&r, p);
	h = FindFirstFileW(r, &data);
	xfree(r);
	if(h == INVALID_HANDLE_VALUE)
		return 0;
	FindClose(h);
	ft = &data.ftLastWriteTime;
	return (Time)ft->dwLowDateTime + ((Time)ft->dwHighDateTime<<32);
}

bool
isabs(char *p)
{
	// c:/ or c:\ at beginning
	if(('A' <= p[0] && p[0] <= 'Z') || ('a' <= p[0] && p[0] <= 'z'))
		return p[1] == ':' && (p[2] == '/' || p[2] == '\\');
	// / or \ at beginning
	return p[0] == '/' || p[0] == '\\';
}

void
readfile(Buf *b, char *file)
{
	HANDLE h;
	Rune *r;

	breset(b);
	if(vflag > 2)
		errprintf("read %s\n", file);
	torune(&r, file);
	h = CreateFileW(r, GENERIC_READ, FILE_SHARE_READ|FILE_SHARE_WRITE, nil, OPEN_EXISTING, 0, 0);
	if(h == INVALID_HANDLE_VALUE)
		fatal("open %s: %s", file, errstr());
	breadfrom(b, h);
	CloseHandle(h);
}

void
writefile(Buf *b, char *file, int exec)
{
	HANDLE h;
	Rune *r;
	DWORD n;

	USED(exec);

	if(vflag > 2)
		errprintf("write %s\n", file);
	torune(&r, file);
	h = CreateFileW(r, GENERIC_WRITE, FILE_SHARE_READ|FILE_SHARE_WRITE, nil, CREATE_ALWAYS, 0, 0);
	if(h == INVALID_HANDLE_VALUE)
		fatal("create %s: %s", file, errstr());
	n = 0;
	if(!WriteFile(h, b->p, b->len, &n, 0))
		fatal("write %s: %s", file, errstr());
	CloseHandle(h);
}
	

void
xmkdir(char *p)
{
	Rune *r;

	torune(&r, p);
	if(!CreateDirectoryW(r, nil))
		fatal("mkdir %s: %s", p, errstr());
	xfree(r);
}

void
xmkdirall(char *p)
{
	int c;
	char *q, *q2;
	
	if(isdir(p))
		return;
	q = strrchr(p, '/');
	q2 = strrchr(p, '\\');
	if(q2 != nil && (q == nil || q < q2))
		q = q2;
	if(q != nil) {
		c = *q;
		*q = '\0';
		xmkdirall(p);
		*q = c;
	}
	xmkdir(p);
}

void
xremove(char *p)
{
	int attr;
	Rune *r;

	torune(&r, p);
	attr = GetFileAttributesW(r);
	if(attr >= 0) {
		if(attr & FILE_ATTRIBUTE_DIRECTORY)
			RemoveDirectoryW(r);
		else
			DeleteFileW(r);
	}
	xfree(r);
}

void
xreaddir(Vec *dst, char *dir)
{
	Rune *r;
	Buf b;
	HANDLE h;
	WIN32_FIND_DATAW data;
	char *p, *q;

	binit(&b);
	vreset(dst);

	bwritestr(&b, dir);
	bwritestr(&b, "\\*");
	torune(&r, bstr(&b));

	h = FindFirstFileW(r, &data);
	xfree(r);
	if(h == INVALID_HANDLE_VALUE)
		goto out;
	do{
		toutf(&b, data.cFileName);
		p = bstr(&b);
		q = xstrrchr(p, '\\');
		if(q != nil)
			p = q+1;
		if(!streq(p, ".") && !streq(p, ".."))
			vadd(dst, p);
	}while(FindNextFileW(h, &data));
	FindClose(h);

out:
	bfree(&b);
}

char*
xworkdir(void)
{
	Rune buf[1024];
	Rune tmp[MAX_PATH];
	Rune go[3] = {'g', 'o', '\0'};
	int n;
	Buf b;

	n = GetTempPathW(nelem(buf), buf);
	if(n <= 0)
		fatal("GetTempPath: %s", errstr());
	buf[n] = '\0';

	if(GetTempFileNameW(buf, go, 0, tmp) == 0)
		fatal("GetTempFileName: %s", errstr());
	DeleteFileW(tmp);
	if(!CreateDirectoryW(tmp, nil))
		fatal("create tempdir: %s", errstr());
	
	binit(&b);
	toutf(&b, tmp);
	return btake(&b);
}

void
xremoveall(char *p)
{
	int i;
	Buf b;
	Vec dir;
	Rune *r;

	binit(&b);
	vinit(&dir);

	torune(&r, p);
	if(isdir(p)) {
		xreaddir(&dir, p);
		for(i=0; i<dir.len; i++) {
			bprintf(&b, "%s/%s", p, dir.p[i]);
			xremoveall(bstr(&b));
		}
		RemoveDirectoryW(r);
	} else {
		DeleteFileW(r);
	}
	xfree(r);
	
	bfree(&b);
	vfree(&dir);	
}

void
fatal(char *msg, ...)
{
	static char buf1[1024];
	va_list arg;

	va_start(arg, msg);
	vsnprintf(buf1, sizeof buf1, msg, arg);
	va_end(arg);

	errprintf("go tool dist: %s\n", buf1);
	
	bgwait();
	ExitProcess(1);
}

// HEAP is the persistent handle to the default process heap.
static HANDLE HEAP = INVALID_HANDLE_VALUE;

void*
xmalloc(int n)
{
	void *p;

	if(HEAP == INVALID_HANDLE_VALUE)
		HEAP = GetProcessHeap();
	p = HeapAlloc(HEAP, 0, n);
	if(p == nil)
		fatal("out of memory allocating %d: %s", n, errstr());
	memset(p, 0, n);
	return p;
}

char*
xstrdup(char *p)
{
	char *q;

	q = xmalloc(strlen(p)+1);
	strcpy(q, p);
	return q;
}

void
xfree(void *p)
{
	if(HEAP == INVALID_HANDLE_VALUE)
		HEAP = GetProcessHeap();
	HeapFree(HEAP, 0, p);
}

void*
xrealloc(void *p, int n)
{
	if(p == nil)
		return xmalloc(n);
	if(HEAP == INVALID_HANDLE_VALUE)
		HEAP = GetProcessHeap();
	p = HeapReAlloc(HEAP, 0, p, n);
	if(p == nil)
		fatal("out of memory reallocating %d", n);
	return p;
}

bool
hassuffix(char *p, char *suffix)
{
	int np, ns;

	np = strlen(p);
	ns = strlen(suffix);
	return np >= ns && streq(p+np-ns, suffix);
}

bool
hasprefix(char *p, char *prefix)
{
	return strncmp(p, prefix, strlen(prefix)) == 0;
}

bool
contains(char *p, char *sep)
{
	return strstr(p, sep) != nil;
}

bool
streq(char *p, char *q)
{
	return strcmp(p, q) == 0;
}

char*
lastelem(char *p)
{
	char *out;

	out = p;
	for(; *p; p++)
		if(*p == '/' || *p == '\\')
			out = p+1;
	return out;
}

void
xmemmove(void *dst, void *src, int n)
{
	memmove(dst, src, n);
}

int
xmemcmp(void *a, void *b, int n)
{
	return memcmp(a, b, n);
}

int
xstrlen(char *p)
{
	return strlen(p);
}

void
xexit(int n)
{
	ExitProcess(n);
}

void
xatexit(void (*f)(void))
{
	atexit(f);
}

void
xprintf(char *fmt, ...)
{
	va_list arg;
	
	va_start(arg, fmt);
	vprintf(fmt, arg);
	va_end(arg);
}

void
errprintf(char *fmt, ...)
{
	va_list arg;
	
	va_start(arg, fmt);
	vfprintf(stderr, fmt, arg);
	va_end(arg);
}

int
main(int argc, char **argv)
{
	SYSTEM_INFO si;

	setvbuf(stdout, nil, _IOLBF, 0);
	setvbuf(stderr, nil, _IOLBF, 0);

	slash = "\\";
	gohostos = "windows";

	GetSystemInfo(&si);
	switch(si.wProcessorArchitecture) {
	case PROCESSOR_ARCHITECTURE_AMD64:
		gohostarch = "amd64";
		break;
	case PROCESSOR_ARCHITECTURE_INTEL:
		gohostarch = "386";
		break;
	default:
		fatal("unknown processor architecture");
	}

	init();

	xmain(argc, argv);
	return 0;
}

void
xqsort(void *data, int n, int elemsize, int (*cmp)(const void*, const void*))
{
	qsort(data, n, elemsize, cmp);
}

int
xstrcmp(char *a, char *b)
{
	return strcmp(a, b);
}

char*
xstrstr(char *a, char *b)
{
	return strstr(a, b);
}

char*
xstrrchr(char *p, int c)
{
	char *ep;
	
	ep = p+strlen(p);
	for(ep=p+strlen(p); ep >= p; ep--)
		if(*ep == c)
			return ep;
	return nil;
}

// xsamefile reports whether f1 and f2 are the same file (or dir)
int
xsamefile(char *f1, char *f2)
{
	Rune *ru;
	HANDLE fd1, fd2;
	BY_HANDLE_FILE_INFORMATION fi1, fi2;
	int r;

	// trivial case
	if(streq(f1, f2))
		return 1;
	
	torune(&ru, f1);
	// refer to ../../os/stat_windows.go:/sameFile
	fd1 = CreateFileW(ru, 0, 0, NULL, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, 0);
	xfree(ru);
	if(fd1 == INVALID_HANDLE_VALUE)
		return 0;
	torune(&ru, f2);
	fd2 = CreateFileW(ru, 0, 0, NULL, OPEN_EXISTING, FILE_FLAG_BACKUP_SEMANTICS, 0);
	xfree(ru);
	if(fd2 == INVALID_HANDLE_VALUE) {
		CloseHandle(fd1);
		return 0;
	}
	r = GetFileInformationByHandle(fd1, &fi1) != 0 && GetFileInformationByHandle(fd2, &fi2) != 0;
	CloseHandle(fd2);
	CloseHandle(fd1);
	if(r != 0 &&
	   fi1.dwVolumeSerialNumber == fi2.dwVolumeSerialNumber &&
	   fi1.nFileIndexHigh == fi2.nFileIndexHigh &&
	   fi1.nFileIndexLow == fi2.nFileIndexLow)
	   	return 1;
	return 0;
}

// xtryexecfunc tries to execute function f, if any illegal instruction
// signal received in the course of executing that function, it will
// return 0, otherwise it will return 1.
int
xtryexecfunc(void (*f)(void))
{
	return 0; // suffice for now
}

static void
cpuid(int dst[4], int ax)
{
	// NOTE: This asm statement is for mingw.
	// If we ever support MSVC, use __cpuid(dst, ax)
	// to use the built-in.
#if defined(__i386__) || defined(__x86_64__)
	asm volatile("cpuid"
		: "=a" (dst[0]), "=b" (dst[1]), "=c" (dst[2]), "=d" (dst[3])
		: "0" (ax));
#else
	dst[0] = dst[1] = dst[2] = dst[3] = 0;
#endif
}

bool
cansse2(void)
{
	int info[4];
	
	cpuid(info, 1);
	return (info[3] & (1<<26)) != 0;	// SSE2
}


#endif // __WINDOWS__
