// Derived from Plan 9 from User Space src/libmach/Linux.c
// http://code.swtch.com/plan9port/src/tip/src/libmach/Linux.c
//
//	Copyright © 1994-1999 Lucent Technologies Inc.
//	Power PC support Copyright © 1995-2004 C H Forsyth (forsyth@terzarima.net).
//	Portions Copyright © 1997-1999 Vita Nuova Limited.
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com).
//	Revisions Copyright © 2000-2004 Lucent Technologies Inc. and others.
//	Portions Copyright © 2001-2007 Russ Cox.
//	Portions Copyright © 2009 The Go Authors.  All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

#include <u.h>
#include <sys/ptrace.h>
#include <sys/signal.h>
#include <errno.h>
#include <libc.h>
#include <bio.h>
#include <mach_amd64.h>
#include <ureg_amd64.h>
#undef waitpid

typedef struct Ureg Ureg;

static Maprw ptracesegrw;
static Maprw ptraceregrw;

// /usr/include/asm-x86_64/user.h
struct user_regs_struct {
	unsigned long r15,r14,r13,r12,rbp,rbx,r11,r10;
	unsigned long r9,r8,rax,rcx,rdx,rsi,rdi,orig_rax;
	unsigned long rip,cs,eflags;
	unsigned long rsp,ss;
  	unsigned long fs_base, gs_base;
	unsigned long ds,es,fs,gs;
};

static int
isstopped(int pid)
{
	char buf[1024];
	int fd, n;
	char *p;

	snprint(buf, sizeof buf, "/proc/%d/stat", pid);
	if((fd = open(buf, OREAD)) < 0)
		return 0;
	n = read(fd, buf, sizeof buf-1);
	close(fd);
	if(n <= 0)
		return 0;
	buf[n] = 0;

	/* command name is in parens, no parens afterward */
	p = strrchr(buf, ')');
	if(p == nil || *++p != ' ')
		return 0;
	++p;

	/* next is state - T is stopped for tracing */
	return *p == 'T';
}

static int
waitstop(int pid)
{
	int p, status;

	if(isstopped(pid))
		return 0;
	for(;;){
		p = waitpid(pid, &status, WUNTRACED|__WALL);
		if(p <= 0){
			if(errno == ECHILD){
				if(isstopped(pid))
					return 0;
			}
			return -1;
		}
		if(WIFEXITED(status) || WIFSTOPPED(status))
			return 0;
	}
}

static int attachedpids[1000];
static int nattached;

static int
ptraceattach(int pid)
{
	int i;

	for(i=0; i<nattached; i++)
		if(attachedpids[i] == pid)
			return 0;
	if(nattached == nelem(attachedpids)){
		werrstr("attached to too many processes");
		return -1;
	}

	if(ptrace(PTRACE_ATTACH, pid, 0, 0) < 0){
		werrstr("ptrace attach %d: %r", pid);
		return -1;
	}
	
	if(waitstop(pid) < 0){
		fprint(2, "waitstop %d: %r", pid);
		ptrace(PTRACE_DETACH, pid, 0, 0);
		return -1;
	}
	attachedpids[nattached++] = pid;
	return 0;
}

Map*
attachproc(int pid, Fhdr *fp)
{
	char buf[64];
	Map *map;
	vlong n;

	if(ptraceattach(pid) < 0)
		return nil;

	map = newmap(0, 4);
	if (!map)
		return 0;
	map->pid = pid;
	if(mach->regsize)
		setmap(map, -1, 0, mach->regsize, 0, "regs", ptraceregrw);
//	if(mach->fpregsize)
//		setmap(map, -1, mach->regsize, mach->regsize+mach->fpregsize, 0, "fpregs", ptraceregrw);
	setmap(map, -1, fp->txtaddr, fp->txtaddr+fp->txtsz, fp->txtaddr, "*text", ptracesegrw);
	setmap(map, -1, fp->dataddr, mach->utop, fp->dataddr, "*data", ptracesegrw);
	return map;
}
	
void
detachproc(Map *m)
{
	if(m->pid > 0)
		ptrace(PTRACE_DETACH, m->pid, 0, 0);
	free(m);
}

/* /proc/pid/stat contains
	pid
	command in parens
	0. state
	1. ppid
	2. pgrp
	3. session
	4. tty_nr
	5. tpgid
	6. flags (math=4, traced=10)
	7. minflt
	8. cminflt
	9. majflt
	10. cmajflt
	11. utime
	12. stime
	13. cutime
	14. cstime
	15. priority
	16. nice
	17. 0
	18. itrealvalue
	19. starttime
	20. vsize
	21. rss
	22. rlim
	23. startcode
	24. endcode
	25. startstack
	26. kstkesp
	27. kstkeip
	28. pending signal bitmap
	29. blocked signal bitmap
	30. ignored signal bitmap
	31. caught signal bitmap
	32. wchan
	33. nswap
	34. cnswap
	35. exit_signal
	36. processor
*/

int
procnotes(int pid, char ***pnotes)
{
	char buf[1024], *f[40];
	int fd, i, n, nf;
	char *p, *s, **notes;
	ulong sigs;
	extern char *_p9sigstr(int, char*);

	*pnotes = nil;
	snprint(buf, sizeof buf, "/proc/%d/stat", pid);
	if((fd = open(buf, OREAD)) < 0){
		fprint(2, "open %s: %r\n", buf);
		return -1;
	}
	n = read(fd, buf, sizeof buf-1);
	close(fd);
	if(n <= 0){
		fprint(2, "read %s: %r\n", buf);
		return -1;
	}
	buf[n] = 0;

	/* command name is in parens, no parens afterward */
	p = strrchr(buf, ')');
	if(p == nil || *++p != ' '){
		fprint(2, "bad format in /proc/%d/stat\n", pid);
		return -1;
	}
	++p;

	nf = tokenize(p, f, nelem(f));
	if(0) print("code 0x%lux-0x%lux stack 0x%lux kstk 0x%lux keip 0x%lux pending 0x%lux\n",
		strtoul(f[23], 0, 0), strtoul(f[24], 0, 0), strtoul(f[25], 0, 0),
		strtoul(f[26], 0, 0), strtoul(f[27], 0, 0), strtoul(f[28], 0, 0));
	if(nf <= 28)
		return -1;

	sigs = strtoul(f[28], 0, 0) & ~(1<<SIGCONT);
	if(sigs == 0){
		*pnotes = nil;
		return 0;
	}

	notes = mallocz(32*sizeof(char*), 0);
	if(notes == nil)
		return -1;
	n = 0;
	for(i=0; i<32; i++){
		if((sigs&(1<<i)) == 0)
			continue;
		if((s = _p9sigstr(i, nil)) == nil)
			continue;
		notes[n++] = s;
	}
	*pnotes = notes;
	return n;
}

int
ctlproc(int pid, char *msg)
{
	int i, p, status;

	if(strcmp(msg, "attached") == 0){
		for(i=0; i<nattached; i++)
			if(attachedpids[i]==pid)
				return 0;
		if(nattached == nelem(attachedpids)){
			werrstr("attached to too many processes");
			return -1;
		}
		attachedpids[nattached++] = pid;
		return 0;
	}

	if(strcmp(msg, "hang") == 0){
		if(pid == getpid())
			return ptrace(PTRACE_TRACEME, 0, 0, 0);
		werrstr("can only hang self");
		return -1;
	}
	if(strcmp(msg, "kill") == 0)
		return ptrace(PTRACE_KILL, pid, 0, 0);
	if(strcmp(msg, "startstop") == 0){
		if(ptrace(PTRACE_CONT, pid, 0, 0) < 0)
			return -1;
		return waitstop(pid);
	}
	if(strcmp(msg, "sysstop") == 0){
		if(ptrace(PTRACE_SYSCALL, pid, 0, 0) < 0)
			return -1;
		return waitstop(pid);
	}
	if(strcmp(msg, "stop") == 0){
		if(kill(pid, SIGSTOP) < 0)
			return -1;
		return waitstop(pid);
	}
	if(strcmp(msg, "step") == 0){
		if(ptrace(PTRACE_SINGLESTEP, pid, 0, 0) < 0)
			return -1;
		return waitstop(pid);
	}
	if(strcmp(msg, "waitstop") == 0)
		return waitstop(pid);
	if(strcmp(msg, "start") == 0)
		return ptrace(PTRACE_CONT, pid, 0, 0);
	werrstr("unknown control message '%s'", msg);
	return -1;
}

char*
proctextfile(int pid)
{
	static char buf[1024], pbuf[128];

	snprint(pbuf, sizeof pbuf, "/proc/%d/exe", pid);
	if(readlink(pbuf, buf, sizeof buf) >= 0)
		return strdup(buf);
	if(access(pbuf, AEXIST) >= 0)
		return strdup(pbuf);
	return nil;
}

int
procthreadpids(int pid, int **thread)
{
	int i, fd, nd, *t, nt;
	char buf[100];
	Dir *d;
	
	snprint(buf, sizeof buf, "/proc/%d/task", pid);
	if((fd = open(buf, OREAD)) < 0)
		return -1;
	nd = dirreadall(fd, &d);
	close(fd);
	if(nd < 0)
		return -1;
	nt = 0;
	for(i=0; i<nd; i++)
		if(d[i].mode&DMDIR)
			nt++;
	t = malloc(nt*sizeof t[0]);
	nt = 0;
	for(i=0; i<nd; i++)
		if(d[i].mode&DMDIR)
			t[nt++] = atoi(d[i].name);
	*thread = t;
	return nt;
}

static int
ptracerw(int type, int xtype, int isr, int pid, uvlong addr, void *v, uint n)
{
	int i;
	uintptr u;
	uchar buf[sizeof(uintptr)];

	for(i=0; i<n; i+=sizeof(uintptr)){
		if(isr){
			errno = 0;
			u = ptrace(type, pid, addr+i, 0);
			if(errno)
				goto ptraceerr;
			if(n-i >= sizeof(uintptr))
				*(uintptr*)((char*)v+i) = u;
			else{
				*(uintptr*)buf = u;
				memmove((char*)v+i, buf, n-i);
			}
		}else{
			if(n-i >= sizeof(uintptr))
				u = *(uintptr*)((char*)v+i);
			else{
				errno = 0;
				u = ptrace(xtype, pid, addr+i, 0);
				if(errno)
					return -1;
				*(uintptr*)buf = u;
				memmove(buf, (char*)v+i, n-i);
				u = *(uintptr*)buf;
			}
			if(ptrace(type, pid, addr+i, u) < 0)
				goto ptraceerr;
		}
	}
	return 0;

ptraceerr:
	werrstr("ptrace %s addr=%#llux pid=%d: %r", isr ? "read" : "write", addr, pid);
	return -1;
}

static int
ptracesegrw(Map *map, Seg *seg, uvlong addr, void *v, uint n, int isr)
{
	return ptracerw(isr ? PTRACE_PEEKDATA : PTRACE_POKEDATA, PTRACE_PEEKDATA,
		isr, map->pid, addr, v, n);
}

static int
go2linux(uvlong addr)
{
	switch(addr){
	case offsetof(Ureg, ax):
		return offsetof(struct user_regs_struct, rax);
	case offsetof(Ureg, bx):
		return offsetof(struct user_regs_struct, rbx);
	case offsetof(Ureg, cx):
		return offsetof(struct user_regs_struct, rcx);
	case offsetof(Ureg, dx):
		return offsetof(struct user_regs_struct, rdx);
	case offsetof(Ureg, si):
		return offsetof(struct user_regs_struct, rsi);
	case offsetof(Ureg, di):
		return offsetof(struct user_regs_struct, rdi);
	case offsetof(Ureg, bp):
		return offsetof(struct user_regs_struct, rbp);
	case offsetof(Ureg, r8):
		return offsetof(struct user_regs_struct, r8);
	case offsetof(Ureg, r9):
		return offsetof(struct user_regs_struct, r9);
	case offsetof(Ureg, r10):
		return offsetof(struct user_regs_struct, r10);
	case offsetof(Ureg, r11):
		return offsetof(struct user_regs_struct, r11);
	case offsetof(Ureg, r12):
		return offsetof(struct user_regs_struct, r12);
	case offsetof(Ureg, r13):
		return offsetof(struct user_regs_struct, r13);
	case offsetof(Ureg, r14):
		return offsetof(struct user_regs_struct, r14);
	case offsetof(Ureg, r15):
		return offsetof(struct user_regs_struct, r15);
	case offsetof(Ureg, ds):
		return offsetof(struct user_regs_struct, ds);
	case offsetof(Ureg, es):
		return offsetof(struct user_regs_struct, es);
	case offsetof(Ureg, fs):
		return offsetof(struct user_regs_struct, fs);
	case offsetof(Ureg, gs):
		return offsetof(struct user_regs_struct, gs);
	case offsetof(Ureg, ip):
		return offsetof(struct user_regs_struct, rip);
	case offsetof(Ureg, cs):
		return offsetof(struct user_regs_struct, cs);
	case offsetof(Ureg, flags):
		return offsetof(struct user_regs_struct, eflags);
	case offsetof(Ureg, sp):
		return offsetof(struct user_regs_struct, rsp);
	case offsetof(Ureg, ss):
		return offsetof(struct user_regs_struct, ss);
	}
	return -1;
}

static int
ptraceregrw(Map *map, Seg *seg, uvlong addr, void *v, uint n, int isr)
{
	int laddr;
	uvlong u;
	
	if((laddr = go2linux(addr)) < 0){
		if(isr){
			memset(v, 0, n);
			return 0;
		}
		werrstr("register %llud not available", addr);
		return -1;
	}
	
	if(isr){
		errno = 0;
		u = ptrace(PTRACE_PEEKUSER, map->pid, laddr, 0);
		if(errno)
			goto ptraceerr;
		switch(n){
		case 1:
			*(uint8*)v = u;
			break;
		case 2:
			*(uint16*)v = u;
			break;
		case 4:
			*(uint32*)v = u;
			break;
		case 8:
			*(uint64*)v = u;
			break;
		default:
			werrstr("bad register size");
			return -1;
		}
	}else{
		switch(n){
		case 1:
			u = *(uint8*)v;
			break;
		case 2:
			u = *(uint16*)v;
			break;
		case 4:
			u = *(uint32*)v;
			break;
		case 8:
			u = *(uint64*)v;
			break;
		default:
			werrstr("bad register size");
			return -1;
		}
		if(ptrace(PTRACE_POKEUSER, map->pid, laddr, (void*)(uintptr)u) < 0)
			goto ptraceerr;
	}
	return 0;

ptraceerr:
	werrstr("ptrace %s register laddr=%d pid=%d: %r", isr ? "read" : "write", laddr, map->pid);
	return -1;	
}
