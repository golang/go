// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

/*
Input to cgo.

GOARCH=amd64 go tool cgo -cdefs defs_darwin.go >defs_darwin_amd64.h
GOARCH=386 go tool cgo -cdefs defs_darwin.go >defs_darwin_386.h
*/

package runtime

/*
#define __DARWIN_UNIX03 0
#include <mach/mach_time.h>
#include <sys/types.h>
#include <sys/time.h>
#include <errno.h>
#include <signal.h>
#include <sys/event.h>
#include <sys/mman.h>
#include <pthread.h>
#include <fcntl.h>
*/
import "C"

const (
	EINTR     = C.EINTR
	EFAULT    = C.EFAULT
	EAGAIN    = C.EAGAIN
	ETIMEDOUT = C.ETIMEDOUT

	PROT_NONE  = C.PROT_NONE
	PROT_READ  = C.PROT_READ
	PROT_WRITE = C.PROT_WRITE
	PROT_EXEC  = C.PROT_EXEC

	MAP_ANON    = C.MAP_ANON
	MAP_PRIVATE = C.MAP_PRIVATE
	MAP_FIXED   = C.MAP_FIXED

	MADV_DONTNEED      = C.MADV_DONTNEED
	MADV_FREE          = C.MADV_FREE
	MADV_FREE_REUSABLE = C.MADV_FREE_REUSABLE
	MADV_FREE_REUSE    = C.MADV_FREE_REUSE

	SA_SIGINFO   = C.SA_SIGINFO
	SA_RESTART   = C.SA_RESTART
	SA_ONSTACK   = C.SA_ONSTACK
	SA_USERTRAMP = C.SA_USERTRAMP
	SA_64REGSET  = C.SA_64REGSET

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
	EV_EOF       = C.EV_EOF
	EVFILT_READ  = C.EVFILT_READ
	EVFILT_WRITE = C.EVFILT_WRITE

	PTHREAD_CREATE_DETACHED = C.PTHREAD_CREATE_DETACHED

	F_SETFD    = C.F_SETFD
	F_GETFL    = C.F_GETFL
	F_SETFL    = C.F_SETFL
	FD_CLOEXEC = C.FD_CLOEXEC

	O_NONBLOCK = C.O_NONBLOCK
)

type StackT C.struct_sigaltstack
type Sighandler C.union___sigaction_u

type Sigaction C.struct___sigaction // used in syscalls
type Usigaction C.struct_sigaction  // used by sigaction second argument
type Sigset C.sigset_t
type Sigval C.union_sigval
type Siginfo C.siginfo_t
type Timeval C.struct_timeval
type Itimerval C.struct_itimerval
type Timespec C.struct_timespec

type FPControl C.struct_fp_control
type FPStatus C.struct_fp_status
type RegMMST C.struct_mmst_reg
type RegXMM C.struct_xmm_reg

type Regs64 C.struct_x86_thread_state64
type FloatState64 C.struct_x86_float_state64
type ExceptionState64 C.struct_x86_exception_state64
type Mcontext64 C.struct_mcontext64

type Regs32 C.struct_i386_thread_state
type FloatState32 C.struct_i386_float_state
type ExceptionState32 C.struct_i386_exception_state
type Mcontext32 C.struct_mcontext32

type Ucontext C.struct_ucontext

type Kevent C.struct_kevent

type Pthread C.pthread_t
type PthreadAttr C.pthread_attr_t
type PthreadMutex C.pthread_mutex_t
type PthreadMutexAttr C.pthread_mutexattr_t
type PthreadCond C.pthread_cond_t
type PthreadCondAttr C.pthread_condattr_t

type MachTimebaseInfo C.mach_timebase_info_data_t
