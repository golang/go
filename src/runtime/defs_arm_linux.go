// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

/*
Input to cgo.
On a Debian Lenny arm linux distribution:

cgo -cdefs defs_arm.c >arm/defs.h
*/

package runtime

/*
#cgo CFLAGS: -I/usr/src/linux-headers-2.6.26-2-versatile/include

#define __ARCH_SI_UID_T int
#include <asm/signal.h>
#include <asm/mman.h>
#include <asm/sigcontext.h>
#include <asm/ucontext.h>
#include <asm/siginfo.h>
#include <linux/time.h>

struct xsiginfo {
	int si_signo;
	int si_errno;
	int si_code;
	char _sifields[4];
};

#undef sa_handler
#undef sa_flags
#undef sa_restorer
#undef sa_mask

struct xsigaction {
	void (*sa_handler)(void);
	unsigned long sa_flags;
	void (*sa_restorer)(void);
	unsigned int sa_mask;		// mask last for extensibility
};
*/
import "C"

const (
	PROT_NONE  = C.PROT_NONE
	PROT_READ  = C.PROT_READ
	PROT_WRITE = C.PROT_WRITE
	PROT_EXEC  = C.PROT_EXEC

	MAP_ANON    = C.MAP_ANONYMOUS
	MAP_PRIVATE = C.MAP_PRIVATE
	MAP_FIXED   = C.MAP_FIXED

	MADV_DONTNEED = C.MADV_DONTNEED

	SA_RESTART  = C.SA_RESTART
	SA_ONSTACK  = C.SA_ONSTACK
	SA_RESTORER = C.SA_RESTORER
	SA_SIGINFO  = C.SA_SIGINFO

	SIGHUP    = C.SIGHUP
	SIGINT    = C.SIGINT
	SIGQUIT   = C.SIGQUIT
	SIGILL    = C.SIGILL
	SIGTRAP   = C.SIGTRAP
	SIGABRT   = C.SIGABRT
	SIGBUS    = C.SIGBUS
	SIGFPE    = C.SIGFPE
	SIGKILL   = C.SIGKILL
	SIGUSR1   = C.SIGUSR1
	SIGSEGV   = C.SIGSEGV
	SIGUSR2   = C.SIGUSR2
	SIGPIPE   = C.SIGPIPE
	SIGALRM   = C.SIGALRM
	SIGSTKFLT = C.SIGSTKFLT
	SIGCHLD   = C.SIGCHLD
	SIGCONT   = C.SIGCONT
	SIGSTOP   = C.SIGSTOP
	SIGTSTP   = C.SIGTSTP
	SIGTTIN   = C.SIGTTIN
	SIGTTOU   = C.SIGTTOU
	SIGURG    = C.SIGURG
	SIGXCPU   = C.SIGXCPU
	SIGXFSZ   = C.SIGXFSZ
	SIGVTALRM = C.SIGVTALRM
	SIGPROF   = C.SIGPROF
	SIGWINCH  = C.SIGWINCH
	SIGIO     = C.SIGIO
	SIGPWR    = C.SIGPWR
	SIGSYS    = C.SIGSYS

	FPE_INTDIV = C.FPE_INTDIV & 0xFFFF
	FPE_INTOVF = C.FPE_INTOVF & 0xFFFF
	FPE_FLTDIV = C.FPE_FLTDIV & 0xFFFF
	FPE_FLTOVF = C.FPE_FLTOVF & 0xFFFF
	FPE_FLTUND = C.FPE_FLTUND & 0xFFFF
	FPE_FLTRES = C.FPE_FLTRES & 0xFFFF
	FPE_FLTINV = C.FPE_FLTINV & 0xFFFF
	FPE_FLTSUB = C.FPE_FLTSUB & 0xFFFF

	BUS_ADRALN = C.BUS_ADRALN & 0xFFFF
	BUS_ADRERR = C.BUS_ADRERR & 0xFFFF
	BUS_OBJERR = C.BUS_OBJERR & 0xFFFF

	SEGV_MAPERR = C.SEGV_MAPERR & 0xFFFF
	SEGV_ACCERR = C.SEGV_ACCERR & 0xFFFF

	ITIMER_REAL    = C.ITIMER_REAL
	ITIMER_PROF    = C.ITIMER_PROF
	ITIMER_VIRTUAL = C.ITIMER_VIRTUAL
)

type Timespec C.struct_timespec
type StackT C.stack_t
type Sigcontext C.struct_sigcontext
type Ucontext C.struct_ucontext
type Timeval C.struct_timeval
type Itimerval C.struct_itimerval
type Siginfo C.struct_xsiginfo
type Sigaction C.struct_xsigaction
