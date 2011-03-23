// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
 * Input to godefs
	godefs -f -m64 defs.c >amd64/defs.h
	godefs -f -m64 defs1.c >>amd64/defs.h
 */

// Linux glibc and Linux kernel define different and conflicting
// definitions for struct sigaction, struct timespec, etc.
// We want the kernel ones, which are in the asm/* headers.
// But then we'd get conflicts when we include the system
// headers for things like ucontext_t, so that happens in
// a separate file, defs1.c.

#include <asm/posix_types.h>
#define size_t __kernel_size_t
#include <asm/signal.h>
#include <asm/siginfo.h>
#include <asm/mman.h>

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
	
	$ITIMER_REAL = ITIMER_REAL,
	$ITIMER_VIRTUAL = ITIMER_VIRTUAL,
	$ITIMER_PROF = ITIMER_PROF,
};

typedef struct timespec $Timespec;
typedef struct timeval $Timeval;
typedef struct sigaction $Sigaction;
typedef siginfo_t $Siginfo;
typedef struct itimerval $Itimerval;
