// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "signals.h"

typedef uint64 __uint64_t;

/* From /usr/include/mach/i386/_structs.h */
#define	_STRUCT_X86_THREAD_STATE64	struct __darwin_x86_thread_state64
_STRUCT_X86_THREAD_STATE64
{
	__uint64_t	__rax;
	__uint64_t	__rbx;
	__uint64_t	__rcx;
	__uint64_t	__rdx;
	__uint64_t	__rdi;
	__uint64_t	__rsi;
	__uint64_t	__rbp;
	__uint64_t	__rsp;
	__uint64_t	__r8;
	__uint64_t	__r9;
	__uint64_t	__r10;
	__uint64_t	__r11;
	__uint64_t	__r12;
	__uint64_t	__r13;
	__uint64_t	__r14;
	__uint64_t	__r15;
	__uint64_t	__rip;
	__uint64_t	__rflags;
	__uint64_t	__cs;
	__uint64_t	__fs;
	__uint64_t	__gs;
};


void
print_thread_state(_STRUCT_X86_THREAD_STATE64* ss)
{
	prints("\nrax     0x");  sys·printpointer((void*)ss->__rax);
	prints("\nrbx     0x");  sys·printpointer((void*)ss->__rbx);
	prints("\nrcx     0x");  sys·printpointer((void*)ss->__rcx);
	prints("\nrdx     0x");  sys·printpointer((void*)ss->__rdx);
	prints("\nrdi     0x");  sys·printpointer((void*)ss->__rdi);
	prints("\nrsi     0x");  sys·printpointer((void*)ss->__rsi);
	prints("\nrbp     0x");  sys·printpointer((void*)ss->__rbp);
	prints("\nrsp     0x");  sys·printpointer((void*)ss->__rsp);
	prints("\nr8      0x");  sys·printpointer((void*)ss->__r8 );
	prints("\nr9      0x");  sys·printpointer((void*)ss->__r9 );
	prints("\nr10     0x");  sys·printpointer((void*)ss->__r10);
	prints("\nr11     0x");  sys·printpointer((void*)ss->__r11);
	prints("\nr12     0x");  sys·printpointer((void*)ss->__r12);
	prints("\nr13     0x");  sys·printpointer((void*)ss->__r13);
	prints("\nr14     0x");  sys·printpointer((void*)ss->__r14);
	prints("\nr15     0x");  sys·printpointer((void*)ss->__r15);
	prints("\nrip     0x");  sys·printpointer((void*)ss->__rip);
	prints("\nrflags  0x");  sys·printpointer((void*)ss->__rflags);
	prints("\ncs      0x");  sys·printpointer((void*)ss->__cs);
	prints("\nfs      0x");  sys·printpointer((void*)ss->__fs);
	prints("\ngs      0x");  sys·printpointer((void*)ss->__gs);
	prints("\n");
}


/* Code generated via: g++ -m64 gen_signals_support.cc && a.out */

static void *adr_at(void *ptr, int32 offs) {
  return (void *)((uint8 *)ptr + offs);
}

static void *ptr_at(void *ptr, int32 offs) {
  return *(void **)((uint8 *)ptr + offs);
}

typedef void ucontext_t;
typedef void _STRUCT_MCONTEXT64;
typedef void _STRUCT_X86_EXCEPTION_STATE64;
typedef void _STRUCT_X86_FLOAT_STATE64;

static _STRUCT_MCONTEXT64 *get_uc_mcontext(ucontext_t *ptr) {
  return (_STRUCT_MCONTEXT64 *)ptr_at(ptr, 48);
}

static _STRUCT_X86_EXCEPTION_STATE64 *get___es(_STRUCT_MCONTEXT64 *ptr) {
  return (_STRUCT_X86_EXCEPTION_STATE64 *)adr_at(ptr, 0);
}

static _STRUCT_X86_THREAD_STATE64 *get___ss(_STRUCT_MCONTEXT64 *ptr) {
  return (_STRUCT_X86_THREAD_STATE64 *)adr_at(ptr, 16);
}

static _STRUCT_X86_FLOAT_STATE64 *get___fs(_STRUCT_MCONTEXT64 *ptr) {
  return (_STRUCT_X86_FLOAT_STATE64 *)adr_at(ptr, 184);
}

/* End of generated code */


/*
 * This assembler routine takes the args from registers, puts them on the stack,
 * and calls sighandler().
 */
extern void sigtramp();

/*
 * Rudimentary reverse-engineered definition of signal interface.
 * You'd think it would be documented.
 */
typedef struct siginfo {
	int32	si_signo;		/* signal number */
	int32	si_errno;		/* errno association */
	int32	si_code;		/* signal code */
	int32	si_pid;			/* sending process */
	int32	si_uid;			/* sender's ruid */
	int32	si_status;		/* exit value */
	void	*si_addr;		/* faulting address */
	/* more stuff here */
} siginfo;


typedef struct  sigaction {
 	union {
		void (*sa_handler)(int32);
		void (*sa_sigaction)(int32, siginfo *, void *);
	} u;				/* signal handler */
	void (*sa_trampoline)(void);	/* kernel callback point; calls sighandler() */
	uint8 sa_mask[4];		/* signal mask to apply */
	int32 sa_flags;			/* see signal options below */
} sigaction;


void
sighandler(int32 sig, siginfo *info, void *context)
{
	if(sig < 0 || sig >= NSIG){
		prints("Signal ");
		sys·printint(sig);
	}else{
		prints(sigtab[sig].name);
	}

        _STRUCT_MCONTEXT64 *uc_mcontext = get_uc_mcontext(context);
        _STRUCT_X86_THREAD_STATE64 *ss = get___ss(uc_mcontext);

	prints("\nFaulting address: 0x");  sys·printpointer(info->si_addr);
	prints("\npc: 0x");  sys·printpointer((void *)ss->__rip);
	prints("\n\n");

	traceback((void *)ss->__rip, (void *)ss->__rsp, (void*)ss->__r15);
	tracebackothers((void*)ss->__r15);
	print_thread_state(ss);

	sys·exit(2);
}


sigaction a;
extern void sigtramp(void);

void
initsig(void)
{
	int32 i;
	a.u.sa_sigaction = (void*)sigtramp;
	a.sa_flags |= 0x40;  /* SA_SIGINFO */
	for(i=0; i<sizeof(a.sa_mask); i++)
		a.sa_mask[i] = 0xFF;
	a.sa_trampoline = sigtramp;

	for(i = 0; i <NSIG; i++)
		if(sigtab[i].catch){
			sys·sigaction(i, &a, (void*)0);
		}
}

static void
unimplemented(int8 *name)
{
	prints(name);
	prints(" not implemented\n");
	*(int32*)1231 = 1231;
}

void
sys·sleep(int64 ms)
{
	unimplemented("sleep");
}

void
lock(Lock *l)
{
	if(cas(&l->key, 0, 1))
		return;
	unimplemented("lock wait");
}

void
unlock(Lock *l)
{
	if(cas(&l->key, 1, 0))
		return;
	unimplemented("unlock wakeup");
}

void
noteclear(Note *n)
{
	n->lock.key = 0;
	lock(&n->lock);
}

void
notesleep(Note *n)
{
	lock(&n->lock);
	unlock(&n->lock);
}

void
notewakeup(Note *n)
{
	unlock(&n->lock);
}

void
newosproc(M *mm, G *gg, void *stk, void (*fn)(void))
{
	unimplemented("newosproc");
}

int32
getprocid(void)
{
	return 0;
}
