// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

/*
Input to cgo.

GOARCH=amd64 go tool cgo -cdefs defs_freebsd.go >defs_freebsd_amd64.h
GOARCH=386 go tool cgo -cdefs defs_freebsd.go >defs_freebsd_386.h
GOARCH=arm go tool cgo -cdefs defs_freebsd.go >defs_freebsd_arm.h
*/

package runtime

/*
#include <sys/types.h>
#include <sys/time.h>
#include <signal.h>
#include <errno.h>
#include <sys/event.h>
#include <sys/mman.h>
#include <sys/ucontext.h>
#include <sys/umtx.h>
#include <sys/rtprio.h>
#include <sys/thr.h>
#include <sys/_sigset.h>
#include <sys/unistd.h>
*/
import "C"

const (
	EINTR  = C.EINTR
	EFAULT = C.EFAULT

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

	UMTX_OP_WAIT_UINT         = C.UMTX_OP_WAIT_UINT
	UMTX_OP_WAIT_UINT_PRIVATE = C.UMTX_OP_WAIT_UINT_PRIVATE
	UMTX_OP_WAKE              = C.UMTX_OP_WAKE
	UMTX_OP_WAKE_PRIVATE      = C.UMTX_OP_WAKE_PRIVATE

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
	EV_RECEIPT   = C.EV_RECEIPT
	EV_ERROR     = C.EV_ERROR
	EVFILT_READ  = C.EVFILT_READ
	EVFILT_WRITE = C.EVFILT_WRITE
)

type Rtprio C.struct_rtprio
type ThrParam C.struct_thr_param
type SigaltstackT C.struct_sigaltstack
type Sigset C.struct___sigset
type StackT C.stack_t

type Siginfo C.siginfo_t

type Mcontext C.mcontext_t
type Ucontext C.ucontext_t

type Timespec C.struct_timespec
type Timeval C.struct_timeval
type Itimerval C.struct_itimerval

type Kevent C.struct_kevent
