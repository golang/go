// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "os_GOOS.h"
#include "arch_GOARCH.h"

int8 *goos = "plan9";
extern SigTab runtime·sigtab[];

int32 runtime·postnote(int32, int8*);

// Called to initialize a new m (including the bootstrap m).
// Called on the parent thread (main thread in case of bootstrap), can allocate memory.
void
runtime·mpreinit(M *mp)
{
	// Initialize stack and goroutine for note handling.
	mp->gsignal = runtime·malg(32*1024);
	mp->notesig = (int8*)runtime·malloc(ERRMAX*sizeof(int8));

	// Initialize stack for handling strings from the
	// errstr system call, as used in package syscall.
	mp->errstr = (byte*)runtime·malloc(ERRMAX*sizeof(byte));
}

// Called to initialize a new m (including the bootstrap m).
// Called on the new thread, can not allocate memory.
void
runtime·minit(void)
{
	// Mask all SSE floating-point exceptions
	// when running on the 64-bit kernel.
	runtime·setfpmasks();
}

// Called from dropm to undo the effect of an minit.
void
runtime·unminit(void)
{
}


static int32
getproccount(void)
{
	int32 fd, i, n, ncpu;
	byte buf[2048];

	fd = runtime·open("/dev/sysstat", OREAD, 0);
	if(fd < 0)
		return 1;
	ncpu = 0;
	for(;;) {
		n = runtime·read(fd, buf, sizeof buf);
		if(n <= 0)
			break;
		for(i = 0; i < n; i++) {
			if(buf[i] == '\n')
				ncpu++;
		}
	}
	runtime·close(fd);
	return ncpu > 0 ? ncpu : 1;
}

static int32
getpid(void)
{
	byte b[20], *c;
	int32 fd;

	runtime·memclr(b, sizeof(b));
	fd = runtime·open("#c/pid", 0, 0);
	if(fd >= 0) {
		runtime·read(fd, b, sizeof(b));
		runtime·close(fd);
	}
	c = b;
	while(*c == ' ' || *c == '\t')
		c++;
	return runtime·atoi(c);
}

void
runtime·osinit(void)
{
	runtime·ncpu = getproccount();
	m->procid = getpid();
	runtime·notify(runtime·sigtramp);
}

void
runtime·crash(void)
{
	runtime·notify(nil);
	*(int32*)0 = 0;
}

void
runtime·get_random_data(byte **rnd, int32 *rnd_len)
{
	*rnd = nil;
	*rnd_len = 0;
}

void
runtime·goenvs(void)
{
}

void
runtime·initsig(void)
{
}

#pragma textflag 7
void
runtime·osyield(void)
{
	runtime·sleep(0);
}

void
runtime·usleep(uint32 µs)
{
	uint32 ms;

	ms = µs/1000;
	if(ms == 0)
		ms = 1;
	runtime·sleep(ms);
}

void
time·now(int64 sec, int32 nsec)
{
	int64 ns;

	ns = runtime·nanotime();
	sec = ns / 1000000000LL;
	nsec = ns - sec * 1000000000LL;
	FLUSH(&sec);
	FLUSH(&nsec);
}

void
runtime·itoa(int32 n, byte *p, uint32 len)
{
	byte *q, c;
	uint32 i;

	if(len <= 1)
		return;

	runtime·memclr(p, len);
	q = p;

	if(n==0) {
		*q++ = '0';
		USED(q);
		return;
	}
	if(n < 0) {
		*q++ = '-';
		p++;
		n = -n;
	}
	for(i=0; n > 0 && i < len; i++) {
		*q++ = '0' + (n%10);
		n = n/10;
	}
	for(q--; q >= p; ) {
		c = *p;
		*p++ = *q;
		*q-- = c;
	}
}

void
runtime·goexitsall(int8 *status)
{
	M *mp;
	int32 pid;

	pid = getpid();
	for(mp=runtime·atomicloadp(&runtime·allm); mp; mp=mp->alllink)
		if(mp->procid != pid)
			runtime·postnote(mp->procid, status);
}

int32
runtime·postnote(int32 pid, int8* msg)
{
	int32 fd, len;
	uint8 buf[128];
	uint8 tmp[16];
	uint8 *p, *q;

	runtime·memclr(buf, sizeof buf);

	/* build path string /proc/pid/note */
	q = tmp;
	p = buf;
	runtime·itoa(pid, tmp, sizeof tmp);
	runtime·memmove((void*)p, (void*)"/proc/", 6);
	for(p += 6; *p++ = *q++; );
	p--;
	runtime·memmove((void*)p, (void*)"/note", 5);

	fd = runtime·open((int8*)buf, OWRITE, 0);
	if(fd < 0)
		return -1;

	len = runtime·findnull((byte*)msg);
	if(runtime·write(fd, msg, len) != len) {
		runtime·close(fd);
		return -1;
	}
	runtime·close(fd);
	return 0;
}

void
runtime·exit(int32 e)
{
	byte tmp[16];
	int8 *status;
 
	if(e == 0)
		status = "";
	else {
		/* build error string */
		runtime·itoa(e, tmp, sizeof tmp);
		status = (int8*)tmp;
	}

	runtime·goexitsall(status);
	runtime·exits(status);
}

void
runtime·newosproc(M *mp, void *stk)
{
	mp->tls[0] = mp->id;	// so 386 asm can find it
	if(0){
		runtime·printf("newosproc stk=%p m=%p g=%p rfork=%p id=%d/%d ostk=%p\n",
			stk, mp, mp->g0, runtime·rfork, mp->id, (int32)mp->tls[0], &mp);
	}

	if(runtime·rfork(RFPROC|RFMEM|RFNOWAIT, stk, mp, mp->g0, runtime·mstart) < 0)
		runtime·throw("newosproc: rfork failed");
}

uintptr
runtime·semacreate(void)
{
	return 1;
}

int32
runtime·semasleep(int64 ns)
{
	int32 ret;
	int32 ms;

	if(ns >= 0) {
		if(ns/1000000 > 0x7fffffffll)
			ms = 0x7fffffff;
		else
			ms = ns/1000000;
		ret = runtime·plan9_tsemacquire(&m->waitsemacount, ms);
		if(ret == 1)
			return 0;  // success
		return -1;  // timeout or interrupted
	}

	while(runtime·plan9_semacquire(&m->waitsemacount, 1) < 0) {
		/* interrupted; try again (c.f. lock_sema.c) */
	}
	return 0;  // success
}

void
runtime·semawakeup(M *mp)
{
	runtime·plan9_semrelease(&mp->waitsemacount, 1);
}

void
os·sigpipe(void)
{
	runtime·throw("too many writes on closed pipe");
}

void
runtime·sigpanic(void)
{
	if(g->sigpc == 0)
		runtime·panicstring("call of nil func value");
	runtime·panicstring(m->notesig);

	if(g->sig == 1 || g->sig == 2)
		runtime·throw("fault");
}

int32
runtime·read(int32 fd, void *buf, int32 nbytes)
{
	return runtime·pread(fd, buf, nbytes, -1LL);
}

int32
runtime·write(int32 fd, void *buf, int32 nbytes)
{
	return runtime·pwrite(fd, buf, nbytes, -1LL);
}

uintptr
runtime·memlimit(void)
{
	return 0;
}

void
runtime·setprof(bool on)
{
	USED(on);
}

static int8 badcallback[] = "runtime: cgo callback on thread not created by Go.\n";

// This runs on a foreign stack, without an m or a g.  No stack split.
#pragma textflag 7
void
runtime·badcallback(void)
{
	runtime·pwrite(2, badcallback, sizeof badcallback - 1, -1LL);
}

static int8 badsignal[] = "runtime: signal received on thread not created by Go.\n";

// This runs on a foreign stack, without an m or a g.  No stack split.
#pragma textflag 7
void
runtime·badsignal(void)
{
	runtime·pwrite(2, badsignal, sizeof badsignal - 1, -1LL);
	runtime·exits(badsignal);
}
