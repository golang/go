// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * Input to godefs
	godefs -carm-gcc -f -I/usr/local/google/src/linux-2.6.28/arch/arm/include -f -I/usr/local/google/src/linux-2.6.28/include -f-D__KERNEL__ -f-D__ARCH_SI_UID_T=int defs_arm.c >arm/defs.h

 * Another input file for ARM defs.h
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

#include <time.h>

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

typedef sigset_t $Sigset;
typedef struct sigaction $Sigaction;
typedef struct timespec $Timespec;
typedef struct sigaltstack $Sigaltstack;
typedef struct sigcontext $Sigcontext;
typedef struct ucontext $Ucontext;

struct xsiginfo {
	int si_signo;
	int si_errno;
	int si_code;
	char _sifields[4];
};

typedef struct xsiginfo $Siginfo;
