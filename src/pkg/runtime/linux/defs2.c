// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * Input to godefs
	godefs -f -m32 \
		-f -I/home/rsc/pub/linux-2.6/arch/x86/include \
		-f -I/home/rsc/pub/linux-2.6/include \
		-f -D_LOOSE_KERNEL_NAMES \
		-f -D__ARCH_SI_UID_T=__kernel_uid32_t \
		defs2.c >386/defs.h

 * The asm header tricks we have to use for Linux on amd64
 * (see defs.c and defs1.c) don't work here, so this is yet another
 * file.  Sigh.
 */

#include <asm/signal.h>
#include <asm/mman.h>
#include <asm/sigcontext.h>
#include <asm/ucontext.h>
#include <asm/siginfo.h>

/*
#include <sys/signal.h>
#include <sys/mman.h>
#include <ucontext.h>
*/

/* This is the sigaction structure from the Linux 2.1.68 kernel which
   is used with the rt_sigaction system call.  For 386 this is not
   defined in any public header file.  */

struct kernel_sigaction {
	__sighandler_t k_sa_handler;
	unsigned long sa_flags;
	void (*sa_restorer) (void);
	sigset_t sa_mask;
};

enum {
	$PROT_NONE = PROT_NONE,
	$PROT_READ = PROT_READ,
	$PROT_WRITE = PROT_WRITE,
	$PROT_EXEC = PROT_EXEC,

	$MAP_ANON = MAP_ANONYMOUS,
	$MAP_PRIVATE = MAP_PRIVATE,
	$MAP_FIXED = MAP_FIXED,

	$SA_RESTART = SA_RESTART,
	$SA_ONSTACK = SA_ONSTACK,
	$SA_RESTORER = SA_RESTORER,
	$SA_SIGINFO = SA_SIGINFO,

	$SIGHUP = SIGHUP,
	$SIGINT = SIGINT,
	$SIGQUIT = SIGQUIT,
	$SIGILL = SIGILL,
	$SIGTRAP = SIGTRAP,
	$SIGABRT = SIGABRT,
	$SIGBUS = SIGBUS,
	$SIGFPE = SIGFPE,
	$SIGKILL = SIGKILL,
	$SIGUSR1 = SIGUSR1,
	$SIGSEGV = SIGSEGV,
	$SIGUSR2 = SIGUSR2,
	$SIGPIPE = SIGPIPE,
	$SIGALRM = SIGALRM,
	$SIGSTKFLT = SIGSTKFLT,
	$SIGCHLD = SIGCHLD,
	$SIGCONT = SIGCONT,
	$SIGSTOP = SIGSTOP,
	$SIGTSTP = SIGTSTP,
	$SIGTTIN = SIGTTIN,
	$SIGTTOU = SIGTTOU,
	$SIGURG = SIGURG,
	$SIGXCPU = SIGXCPU,
	$SIGXFSZ = SIGXFSZ,
	$SIGVTALRM = SIGVTALRM,
	$SIGPROF = SIGPROF,
	$SIGWINCH = SIGWINCH,
	$SIGIO = SIGIO,
	$SIGPWR = SIGPWR,
	$SIGSYS = SIGSYS,
	
	$FPE_INTDIV = FPE_INTDIV,
	$FPE_INTOVF = FPE_INTOVF,
	$FPE_FLTDIV = FPE_FLTDIV,
	$FPE_FLTOVF = FPE_FLTOVF,
	$FPE_FLTUND = FPE_FLTUND,
	$FPE_FLTRES = FPE_FLTRES,
	$FPE_FLTINV = FPE_FLTINV,
	$FPE_FLTSUB = FPE_FLTSUB,
	
	$BUS_ADRALN = BUS_ADRALN,
	$BUS_ADRERR = BUS_ADRERR,
	$BUS_OBJERR = BUS_OBJERR,
	
	$SEGV_MAPERR = SEGV_MAPERR,
	$SEGV_ACCERR = SEGV_ACCERR,
};

typedef struct _fpreg $Fpreg;
typedef struct _fpxreg $Fpxreg;
typedef struct _xmmreg $Xmmreg;
typedef struct _fpstate $Fpstate;
typedef struct timespec $Timespec;
typedef struct timeval $Timeval;
typedef struct kernel_sigaction $Sigaction;
typedef siginfo_t $Siginfo;
typedef struct sigaltstack $Sigaltstack;
typedef struct sigcontext $Sigcontext;
typedef struct ucontext $Ucontext;

