// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

/*
 * Input to cgo -cdefs

GOARCH=386 go tool cgo -cdefs defs2_linux.go >defs_linux_386.h

The asm header tricks we have to use for Linux on amd64
(see defs.c and defs1.c) don't work here, so this is yet another
file.  Sigh.
*/

package runtime

/*
#cgo CFLAGS: -I/tmp/linux/arch/x86/include -I/tmp/linux/include -D_LOOSE_KERNEL_NAMES -D__ARCH_SI_UID_T=__kernel_uid32_t

#define size_t __kernel_size_t
#define pid_t int
#include <asm/signal.h>
#include <asm/mman.h>
#include <asm/sigcontext.h>
#include <asm/ucontext.h>
#include <asm/siginfo.h>
#include <asm-generic/errno.h>
#include <asm-generic/fcntl.h>
#include <asm-generic/poll.h>
#include <linux/eventpoll.h>

// This is the sigaction structure from the Linux 2.1.68 kernel which
//   is used with the rt_sigaction system call. For 386 this is not
//   defined in any public header file.

struct kernel_sigaction {
	__sighandler_t k_sa_handler;
	unsigned long sa_flags;
	void (*sa_restorer) (void);
	unsigned long long sa_mask;
};
*/
import "C"

const (
	EINTR  = C.EINTR
	EAGAIN = C.EAGAIN
	ENOMEM = C.ENOMEM

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

	FPE_INTDIV = C.FPE_INTDIV
	FPE_INTOVF = C.FPE_INTOVF
	FPE_FLTDIV = C.FPE_FLTDIV
	FPE_FLTOVF = C.FPE_FLTOVF
	FPE_FLTUND = C.FPE_FLTUND
	FPE_FLTRES = C.FPE_FLTRES
	FPE_FLTINV = C.FPE_FLTINV
	FPE_FLTSUB = C.FPE_FLTSUB

	BUS_ADRALN = C.BUS_ADRALN
	BUS_ADRERR = C.BUS_ADRERR
	BUS_OBJERR = C.BUS_OBJERR

	SEGV_MAPERR = C.SEGV_MAPERR
	SEGV_ACCERR = C.SEGV_ACCERR

	ITIMER_REAL    = C.ITIMER_REAL
	ITIMER_VIRTUAL = C.ITIMER_VIRTUAL
	ITIMER_PROF    = C.ITIMER_PROF

	O_RDONLY  = C.O_RDONLY
	O_CLOEXEC = C.O_CLOEXEC

	EPOLLIN       = C.POLLIN
	EPOLLOUT      = C.POLLOUT
	EPOLLERR      = C.POLLERR
	EPOLLHUP      = C.POLLHUP
	EPOLLRDHUP    = C.POLLRDHUP
	EPOLLET       = C.EPOLLET
	EPOLL_CLOEXEC = C.EPOLL_CLOEXEC
	EPOLL_CTL_ADD = C.EPOLL_CTL_ADD
	EPOLL_CTL_DEL = C.EPOLL_CTL_DEL
	EPOLL_CTL_MOD = C.EPOLL_CTL_MOD
)

type Fpreg C.struct__fpreg
type Fpxreg C.struct__fpxreg
type Xmmreg C.struct__xmmreg
type Fpstate C.struct__fpstate
type Timespec C.struct_timespec
type Timeval C.struct_timeval
type Sigaction C.struct_kernel_sigaction
type Siginfo C.siginfo_t
type SigaltstackT C.struct_sigaltstack
type Sigcontext C.struct_sigcontext
type Ucontext C.struct_ucontext
type Itimerval C.struct_itimerval
type EpollEvent C.struct_epoll_event
