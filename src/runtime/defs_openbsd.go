// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

/*
Input to cgo.

GOARCH=amd64 go tool cgo -godefs defs_openbsd.go
GOARCH=386 go tool cgo -godefs defs_openbsd.go
GOARCH=arm go tool cgo -godefs defs_openbsd.go
GOARCH=arm64 go tool cgo -godefs defs_openbsd.go
GOARCH=mips64 go tool cgo -godefs defs_openbsd.go
*/

package runtime

/*
#include <sys/types.h>
#include <sys/event.h>
#include <sys/mman.h>
#include <sys/time.h>
#include <sys/unistd.h>
#include <sys/signal.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <signal.h>
*/
import "C"

const (
	EINTR     = C.EINTR
	EFAULT    = C.EFAULT
	EAGAIN    = C.EAGAIN
	ETIMEDOUT = C.ETIMEDOUT

	O_NONBLOCK = C.O_NONBLOCK
	O_CLOEXEC  = C.O_CLOEXEC

	PROT_NONE  = C.PROT_NONE
	PROT_READ  = C.PROT_READ
	PROT_WRITE = C.PROT_WRITE
	PROT_EXEC  = C.PROT_EXEC

	MAP_ANON    = C.MAP_ANON
	MAP_PRIVATE = C.MAP_PRIVATE
	MAP_FIXED   = C.MAP_FIXED
	MAP_STACK   = C.MAP_STACK

	MADV_DONTNEED = C.MADV_DONTNEED
	MADV_FREE     = C.MADV_FREE

	SA_SIGINFO = C.SA_SIGINFO
	SA_RESTART = C.SA_RESTART
	SA_ONSTACK = C.SA_ONSTACK

	PTHREAD_CREATE_DETACHED = C.PTHREAD_CREATE_DETACHED

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
	SIGINFO   = C.SIGINFO
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

	EV_ADD       = C.EV_ADD
	EV_DELETE    = C.EV_DELETE
	EV_CLEAR     = C.EV_CLEAR
	EV_ERROR     = C.EV_ERROR
	EV_EOF       = C.EV_EOF
	EVFILT_READ  = C.EVFILT_READ
	EVFILT_WRITE = C.EVFILT_WRITE
)

type TforkT C.struct___tfork

type Sigcontext C.struct_sigcontext
type Siginfo C.siginfo_t
type Sigset C.sigset_t
type Sigval C.union_sigval

type StackT C.stack_t

type Timespec C.struct_timespec
type Timeval C.struct_timeval
type Itimerval C.struct_itimerval

type KeventT C.struct_kevent

type Pthread C.pthread_t
type PthreadAttr C.pthread_attr_t
type PthreadCond C.pthread_cond_t
type PthreadCondAttr C.pthread_condattr_t
type PthreadMutex C.pthread_mutex_t
type PthreadMutexAttr C.pthread_mutexattr_t
