// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * Input to godefs
 * On a Debian Lenny arm linux distribution:
	godefs -f-I/usr/src/linux-headers-2.6.26-2-versatile/include defs_arm.c
 */

#define __ARCH_SI_UID_T int

#include <asm/signal.h>
#include <asm/mman.h>
#include <asm/sigcontext.h>
#include <asm/ucontext.h>
#include <asm/siginfo.h>
#include <linux/time.h>

/*
#include <sys/signal.h>
#include <sys/mman.h>
#include <ucontext.h>
*/

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

	$FPE_INTDIV = FPE_INTDIV & 0xFFFF,
	$FPE_INTOVF = FPE_INTOVF & 0xFFFF,
	$FPE_FLTDIV = FPE_FLTDIV & 0xFFFF,
	$FPE_FLTOVF = FPE_FLTOVF & 0xFFFF,
	$FPE_FLTUND = FPE_FLTUND & 0xFFFF,
	$FPE_FLTRES = FPE_FLTRES & 0xFFFF,
	$FPE_FLTINV = FPE_FLTINV & 0xFFFF,
	$FPE_FLTSUB = FPE_FLTSUB & 0xFFFF,
	
	$BUS_ADRALN = BUS_ADRALN & 0xFFFF,
	$BUS_ADRERR = BUS_ADRERR & 0xFFFF,
	$BUS_OBJERR = BUS_OBJERR & 0xFFFF,
	
	$SEGV_MAPERR = SEGV_MAPERR & 0xFFFF,
	$SEGV_ACCERR = SEGV_ACCERR & 0xFFFF,

	$ITIMER_REAL = ITIMER_REAL,
	$ITIMER_PROF = ITIMER_PROF,
	$ITIMER_VIRTUAL = ITIMER_VIRTUAL,
};

typedef sigset_t $Sigset;
typedef struct timespec $Timespec;
typedef struct sigaltstack $Sigaltstack;
typedef struct sigcontext $Sigcontext;
typedef struct ucontext $Ucontext;
typedef struct timeval $Timeval;
typedef struct itimerval $Itimerval;

struct xsiginfo {
	int si_signo;
	int si_errno;
	int si_code;
	char _sifields[4];
};

typedef struct xsiginfo $Siginfo;

#undef sa_handler
#undef sa_flags
#undef sa_restorer
#undef sa_mask

struct xsigaction {
	void (*sa_handler)(void);
	unsigned long sa_flags;
	void (*sa_restorer)(void);
	unsigned int sa_mask;		/* mask last for extensibility */
};

typedef struct xsigaction $Sigaction;
