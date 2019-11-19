// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

/*
Input to cgo.

GOARCH=amd64 go tool cgo -cdefs defs_solaris.go >defs_solaris_amd64.h
*/

package runtime

/*
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/select.h>
#include <sys/siginfo.h>
#include <sys/signal.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <sys/ucontext.h>
#include <sys/regset.h>
#include <sys/unistd.h>
#include <sys/fork.h>
#include <sys/port.h>
#include <semaphore.h>
#include <errno.h>
#include <signal.h>
#include <pthread.h>
#include <netdb.h>
*/
import "C"

const (
	EINTR       = C.EINTR
	EBADF       = C.EBADF
	EFAULT      = C.EFAULT
	EAGAIN      = C.EAGAIN
	EBUSY       = C.EBUSY
	ETIME       = C.ETIME
	ETIMEDOUT   = C.ETIMEDOUT
	EWOULDBLOCK = C.EWOULDBLOCK
	EINPROGRESS = C.EINPROGRESS
	ENOSYS      = C.ENOSYS

	PROT_NONE  = C.PROT_NONE
	PROT_READ  = C.PROT_READ
	PROT_WRITE = C.PROT_WRITE
	PROT_EXEC  = C.PROT_EXEC

	MAP_ANON    = C.MAP_ANON
	MAP_PRIVATE = C.MAP_PRIVATE
	MAP_FIXED   = C.MAP_FIXED

	MADV_FREE = C.MADV_FREE

	SA_SIGINFO = C.SA_SIGINFO
	SA_RESTART = C.SA_RESTART
	SA_ONSTACK = C.SA_ONSTACK

	SIGHUP    = C.SIGHUP
	SIGINT    = C.SIGINT
	SIGQUIT   = C.SIGQUIT
	SIGILL    = C.SIGILL
	SIGTRAP   = C.SIGTRAP
	SIGABRT   = C.SIGABRT
	SIGEMT    = C.SIGEMT
	SIGFPE    = C.SIGFPE
	SIGKILL   = C.SIGKILL
	SIGBUS    = C.SIGBUS
	SIGSEGV   = C.SIGSEGV
	SIGSYS    = C.SIGSYS
	SIGPIPE   = C.SIGPIPE
	SIGALRM   = C.SIGALRM
	SIGTERM   = C.SIGTERM
	SIGURG    = C.SIGURG
	SIGSTOP   = C.SIGSTOP
	SIGTSTP   = C.SIGTSTP
	SIGCONT   = C.SIGCONT
	SIGCHLD   = C.SIGCHLD
	SIGTTIN   = C.SIGTTIN
	SIGTTOU   = C.SIGTTOU
	SIGIO     = C.SIGIO
	SIGXCPU   = C.SIGXCPU
	SIGXFSZ   = C.SIGXFSZ
	SIGVTALRM = C.SIGVTALRM
	SIGPROF   = C.SIGPROF
	SIGWINCH  = C.SIGWINCH
	SIGUSR1   = C.SIGUSR1
	SIGUSR2   = C.SIGUSR2

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

	_SC_NPROCESSORS_ONLN = C._SC_NPROCESSORS_ONLN

	PTHREAD_CREATE_DETACHED = C.PTHREAD_CREATE_DETACHED

	FORK_NOSIGCHLD = C.FORK_NOSIGCHLD
	FORK_WAITPID   = C.FORK_WAITPID

	MAXHOSTNAMELEN = C.MAXHOSTNAMELEN

	O_NONBLOCK = C.O_NONBLOCK
	O_CLOEXEC  = C.O_CLOEXEC
	FD_CLOEXEC = C.FD_CLOEXEC
	F_GETFL    = C.F_GETFL
	F_SETFL    = C.F_SETFL
	F_SETFD    = C.F_SETFD

	POLLIN  = C.POLLIN
	POLLOUT = C.POLLOUT
	POLLHUP = C.POLLHUP
	POLLERR = C.POLLERR

	PORT_SOURCE_FD    = C.PORT_SOURCE_FD
	PORT_SOURCE_ALERT = C.PORT_SOURCE_ALERT
	PORT_ALERT_UPDATE = C.PORT_ALERT_UPDATE
)

type SemT C.sem_t

type Sigset C.sigset_t
type StackT C.stack_t

type Siginfo C.siginfo_t
type Sigaction C.struct_sigaction

type Fpregset C.fpregset_t
type Mcontext C.mcontext_t
type Ucontext C.ucontext_t

type Timespec C.struct_timespec
type Timeval C.struct_timeval
type Itimerval C.struct_itimerval

type PortEvent C.port_event_t
type Pthread C.pthread_t
type PthreadAttr C.pthread_attr_t

// depends on Timespec, must appear below
type Stat C.struct_stat
