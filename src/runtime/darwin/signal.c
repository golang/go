// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "runtime.h"
#include "defs.h"
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
	prints("\nrax     ");  sys·printhex(ss->__rax);
	prints("\nrbx     ");  sys·printhex(ss->__rbx);
	prints("\nrcx     ");  sys·printhex(ss->__rcx);
	prints("\nrdx     ");  sys·printhex(ss->__rdx);
	prints("\nrdi     ");  sys·printhex(ss->__rdi);
	prints("\nrsi     ");  sys·printhex(ss->__rsi);
	prints("\nrbp     ");  sys·printhex(ss->__rbp);
	prints("\nrsp     ");  sys·printhex(ss->__rsp);
	prints("\nr8      ");  sys·printhex(ss->__r8 );
	prints("\nr9      ");  sys·printhex(ss->__r9 );
	prints("\nr10     ");  sys·printhex(ss->__r10);
	prints("\nr11     ");  sys·printhex(ss->__r11);
	prints("\nr12     ");  sys·printhex(ss->__r12);
	prints("\nr13     ");  sys·printhex(ss->__r13);
	prints("\nr14     ");  sys·printhex(ss->__r14);
	prints("\nr15     ");  sys·printhex(ss->__r15);
	prints("\nrip     ");  sys·printhex(ss->__rip);
	prints("\nrflags  ");  sys·printhex(ss->__rflags);
	prints("\ncs      ");  sys·printhex(ss->__cs);
	prints("\nfs      ");  sys·printhex(ss->__fs);
	prints("\ngs      ");  sys·printhex(ss->__gs);
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
 * and calls the registered handler.
 */
extern void sigtramp(void);
/*
 * Rudimentary reverse-engineered definition of signal interface.
 * You'd think it would be documented.
 */
struct siginfo {
	int32	si_signo;		/* signal number */
	int32	si_errno;		/* errno association */
	int32	si_code;		/* signal code */
	int32	si_pid;			/* sending process */
	int32	si_uid;			/* sender's ruid */
	int32	si_status;		/* exit value */
	void	*si_addr;		/* faulting address */
	/* more stuff here */
};

struct sigaction {
	void (*sa_handler)(int32, struct siginfo*, void*);	// actual handler
	void (*sa_trampoline)(void);	// assembly trampoline
	uint32 sa_mask;		// signal mask during handler
	int32 sa_flags;			// flags below
};

void
sighandler(int32 sig, struct siginfo *info, void *context)
{
	if(panicking)	// traceback already printed
		sys_Exit(2);
	panicking = 1;

        _STRUCT_MCONTEXT64 *uc_mcontext = get_uc_mcontext(context);
        _STRUCT_X86_THREAD_STATE64 *ss = get___ss(uc_mcontext);

	if(sig < 0 || sig >= NSIG){
		prints("Signal ");
		sys·printint(sig);
	}else{
		prints(sigtab[sig].name);
	}

	prints("\nFaulting address: ");  sys·printpointer(info->si_addr);
	prints("\npc: ");  sys·printhex(ss->__rip);
	prints("\n\n");

	if(gotraceback()){
		traceback((void *)ss->__rip, (void *)ss->__rsp, (void*)ss->__r15);
		tracebackothers((void*)ss->__r15);
		print_thread_state(ss);
	}

	sys_Exit(2);
}

void
sigignore(int32, struct siginfo*, void*)
{
}

struct stack_t {
	byte *sp;
	int64 size;
	int32 flags;
};

void
signalstack(byte *p, int32 n)
{
	struct stack_t st;

	st.sp = p;
	st.size = n;
	st.flags = 0;
	sigaltstack(&st, nil);
}

void	sigaction(int64, void*, void*);

enum {
	SA_SIGINFO = 0x40,
	SA_RESTART = 0x02,
	SA_ONSTACK = 0x01,
	SA_USERTRAMP = 0x100,
	SA_64REGSET = 0x200,
};

void
initsig(void)
{
	int32 i;
	static struct sigaction sa;

	sa.sa_flags |= SA_SIGINFO|SA_ONSTACK;
	sa.sa_mask = 0; // 0xFFFFFFFFU;
	sa.sa_trampoline = sigtramp;
	for(i = 0; i<NSIG; i++) {
		if(sigtab[i].flags) {
			if(sigtab[i].flags & SigCatch) {
				sa.sa_handler = sighandler;
			} else {
				sa.sa_handler = sigignore;
			}
			if(sigtab[i].flags & SigRestart)
				sa.sa_flags |= SA_RESTART;
			else
				sa.sa_flags &= ~SA_RESTART;
			sigaction(i, &sa, nil);
		}
	}
}

static void
unimplemented(int8 *name)
{
	prints(name);
	prints(" not implemented\n");
	*(int32*)1231 = 1231;
}

// Thread-safe allocation of a semaphore.
// Psema points at a kernel semaphore key.
// It starts out zero, meaning no semaphore.
// Fill it in, being careful of others calling initsema
// simultaneously.
static void
initsema(uint32 *psema)
{
	uint32 sema;

	if(*psema != 0)	// already have one
		return;

	sema = mach_semcreate();
	if(!cas(psema, 0, sema)){
		// Someone else filled it in.  Use theirs.
		mach_semdestroy(sema);
		return;
	}
}


// Atomic add and return new value.
static uint32
xadd(uint32 volatile *val, int32 delta)
{
	uint32 oval, nval;

	for(;;){
		oval = *val;
		nval = oval + delta;
		if(cas(val, oval, nval))
			return nval;
	}
}

