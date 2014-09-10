// Copyright 2010 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "os_GOOS.h"
#include "arch_GOARCH.h"
#include "textflag.h"
#include "malloc.h"

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
	mp->gsignal->m = mp;
	mp->notesig = (int8*)runtime·mallocgc(ERRMAX*sizeof(int8), nil, FlagNoScan);

	// Initialize stack for handling strings from the
	// errstr system call, as used in package syscall.
	mp->errstr = (byte*)runtime·mallocgc(ERRMAX*sizeof(byte), nil, FlagNoScan);
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
	g->m->procid = getpid();
	runtime·notify(runtime·sigtramp);
}

void
runtime·crash(void)
{
	runtime·notify(nil);
	*(int32*)0 = 0;
}

#pragma textflag NOSPLIT
void
runtime·get_random_data(byte **rnd, int32 *rnd_len)
{
	static byte random_data[HashRandomBytes];
	int32 fd;

	fd = runtime·open("/dev/random", 0 /* O_RDONLY */, 0);
	if(runtime·read(fd, random_data, HashRandomBytes) == HashRandomBytes) {
		*rnd = random_data;
		*rnd_len = HashRandomBytes;
	} else {
		*rnd = nil;
		*rnd_len = 0;
	}
	runtime·close(fd);
}

void
runtime·goenvs(void)
{
}

void
runtime·initsig(void)
{
}

#pragma textflag NOSPLIT
void
runtime·osyield(void)
{
	runtime·sleep(0);
}

#pragma textflag NOSPLIT
void
runtime·usleep(uint32 µs)
{
	uint32 ms;

	ms = µs/1000;
	if(ms == 0)
		ms = 1;
	runtime·sleep(ms);
}

#pragma textflag NOSPLIT
int64
runtime·nanotime(void)
{
	int64 ns, scratch;

	ns = runtime·nsec(&scratch);
	// TODO(aram): remove hack after I fix _nsec in the pc64 kernel.
	if(ns == 0)
		return scratch;
	return ns;
}

#pragma textflag NOSPLIT
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
	int8 buf[ERRMAX];
	M *mp;
	int32 pid;

	runtime·snprintf((byte*)buf, sizeof buf, "go: exit %s", status);
	pid = getpid();
	for(mp=runtime·atomicloadp(&runtime·allm); mp; mp=mp->alllink)
		if(mp->procid != pid)
			runtime·postnote(mp->procid, buf);
}

int32
runtime·postnote(int32 pid, int8* msg)
{
	int32 fd;
	intgo len;
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

static void exit(void);

#pragma textflag NOSPLIT
void
runtime·exit(int32 e)
{
	void (*fn)(void);

	g->m->scalararg[0] = e;
	fn = exit;
	runtime·onM(&fn);
}

static void
exit(void)
{
	int32 e;
	byte tmp[16];
	int8 *status;
 
 	e = g->m->scalararg[0];
 	g->m->scalararg[0] = 0;

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
	int32 pid;

	if(0)
		runtime·printf("newosproc mp=%p ostk=%p\n", mp, &mp);

	USED(stk);
	if((pid = runtime·rfork(RFPROC|RFMEM|RFNOWAIT)) < 0)
		runtime·throw("newosproc: rfork failed\n");
	if(pid == 0)
		runtime·tstart_plan9(mp);
}

#pragma textflag NOSPLIT
uintptr
runtime·semacreate(void)
{
	return 1;
}

#pragma textflag NOSPLIT
int32
runtime·semasleep(int64 ns)
{
	int32 ret;
	int32 ms;

	if(ns >= 0) {
		ms = runtime·timediv(ns, 1000000, nil);
		if(ms == 0)
			ms = 1;
		ret = runtime·plan9_tsemacquire(&g->m->waitsemacount, ms);
		if(ret == 1)
			return 0;  // success
		return -1;  // timeout or interrupted
	}

	while(runtime·plan9_semacquire(&g->m->waitsemacount, 1) < 0) {
		/* interrupted; try again (c.f. lock_sema.c) */
	}
	return 0;  // success
}

#pragma textflag NOSPLIT
void
runtime·semawakeup(M *mp)
{
	runtime·plan9_semrelease(&mp->waitsemacount, 1);
}

#pragma textflag NOSPLIT
int32
runtime·read(int32 fd, void *buf, int32 nbytes)
{
	return runtime·pread(fd, buf, nbytes, -1LL);
}

#pragma textflag NOSPLIT
int32
runtime·write(uintptr fd, void *buf, int32 nbytes)
{
	return runtime·pwrite((int32)fd, buf, nbytes, -1LL);
}

uintptr
runtime·memlimit(void)
{
	return 0;
}

#pragma dataflag NOPTR
static int8 badsignal[] = "runtime: signal received on thread not created by Go.\n";

// This runs on a foreign stack, without an m or a g.  No stack split.
#pragma textflag NOSPLIT
void
runtime·badsignal2(void)
{
	runtime·pwrite(2, badsignal, sizeof badsignal - 1, -1LL);
	runtime·exits(badsignal);
}
