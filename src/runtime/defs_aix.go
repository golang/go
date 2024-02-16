// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build ignore

/*
Input to cgo -godefs
GOARCH=ppc64 go tool cgo -godefs defs_aix.go > defs_aix_ppc64_tmp.go

This is only a helper to create defs_aix_ppc64.go
Go runtime functions require the "linux" name of fields (ss_sp, si_addr, etc)
However, AIX structures don't provide such names and must be modified.

TODO(aix): create a script to automatise defs_aix creation.

Modifications made:
 - sigset replaced by a [4]uint64 array
 - add sigset_all variable
 - siginfo.si_addr uintptr instead of *byte
 - add (*timeval) set_usec
 - stackt.ss_sp uintptr instead of *byte
 - stackt.ss_size uintptr instead of uint64
 - sigcontext.sc_jmpbuf context64 instead of jumbuf
 - ucontext.__extctx is a uintptr because we don't need extctx struct
 - ucontext.uc_mcontext: replace jumbuf structure by context64 structure
 - sigaction.sa_handler represents union field as both are uintptr
 - tstate.* replace *byte by uintptr


*/

package runtime

/*

#include <sys/types.h>
#include <sys/errno.h>
#include <sys/time.h>
#include <sys/signal.h>
#include <sys/mman.h>
#include <sys/thread.h>
#include <sys/resource.h>

#include <unistd.h>
#include <fcntl.h>
#include <pthread.h>
#include <semaphore.h>
*/
import "C"

const (
	_EPERM     = C.EPERM
	_ENOENT    = C.ENOENT
	_EINTR     = C.EINTR
	_EAGAIN    = C.EAGAIN
	_ENOMEM    = C.ENOMEM
	_EACCES    = C.EACCES
	_EFAULT    = C.EFAULT
	_EINVAL    = C.EINVAL
	_ETIMEDOUT = C.ETIMEDOUT

	_PROT_NONE  = C.PROT_NONE
	_PROT_READ  = C.PROT_READ
	_PROT_WRITE = C.PROT_WRITE
	_PROT_EXEC  = C.PROT_EXEC

	_MAP_ANON      = C.MAP_ANONYMOUS
	_MAP_PRIVATE   = C.MAP_PRIVATE
	_MAP_FIXED     = C.MAP_FIXED
	_MADV_DONTNEED = C.MADV_DONTNEED

	_SIGHUP     = C.SIGHUP
	_SIGINT     = C.SIGINT
	_SIGQUIT    = C.SIGQUIT
	_SIGILL     = C.SIGILL
	_SIGTRAP    = C.SIGTRAP
	_SIGABRT    = C.SIGABRT
	_SIGBUS     = C.SIGBUS
	_SIGFPE     = C.SIGFPE
	_SIGKILL    = C.SIGKILL
	_SIGUSR1    = C.SIGUSR1
	_SIGSEGV    = C.SIGSEGV
	_SIGUSR2    = C.SIGUSR2
	_SIGPIPE    = C.SIGPIPE
	_SIGALRM    = C.SIGALRM
	_SIGCHLD    = C.SIGCHLD
	_SIGCONT    = C.SIGCONT
	_SIGSTOP    = C.SIGSTOP
	_SIGTSTP    = C.SIGTSTP
	_SIGTTIN    = C.SIGTTIN
	_SIGTTOU    = C.SIGTTOU
	_SIGURG     = C.SIGURG
	_SIGXCPU    = C.SIGXCPU
	_SIGXFSZ    = C.SIGXFSZ
	_SIGVTALRM  = C.SIGVTALRM
	_SIGPROF    = C.SIGPROF
	_SIGWINCH   = C.SIGWINCH
	_SIGIO      = C.SIGIO
	_SIGPWR     = C.SIGPWR
	_SIGSYS     = C.SIGSYS
	_SIGTERM    = C.SIGTERM
	_SIGEMT     = C.SIGEMT
	_SIGWAITING = C.SIGWAITING

	_FPE_INTDIV = C.FPE_INTDIV
	_FPE_INTOVF = C.FPE_INTOVF
	_FPE_FLTDIV = C.FPE_FLTDIV
	_FPE_FLTOVF = C.FPE_FLTOVF
	_FPE_FLTUND = C.FPE_FLTUND
	_FPE_FLTRES = C.FPE_FLTRES
	_FPE_FLTINV = C.FPE_FLTINV
	_FPE_FLTSUB = C.FPE_FLTSUB

	_BUS_ADRALN = C.BUS_ADRALN
	_BUS_ADRERR = C.BUS_ADRERR
	_BUS_OBJERR = C.BUS_OBJERR

	_SEGV_MAPERR = C.SEGV_MAPERR
	_SEGV_ACCERR = C.SEGV_ACCERR

	_ITIMER_REAL    = C.ITIMER_REAL
	_ITIMER_VIRTUAL = C.ITIMER_VIRTUAL
	_ITIMER_PROF    = C.ITIMER_PROF

	_O_RDONLY   = C.O_RDONLY
	_O_WRONLY   = C.O_WRONLY
	_O_NONBLOCK = C.O_NONBLOCK
	_O_CREAT    = C.O_CREAT
	_O_TRUNC    = C.O_TRUNC

	_SS_DISABLE  = C.SS_DISABLE
	_SI_USER     = C.SI_USER
	_SIG_BLOCK   = C.SIG_BLOCK
	_SIG_UNBLOCK = C.SIG_UNBLOCK
	_SIG_SETMASK = C.SIG_SETMASK

	_SA_SIGINFO = C.SA_SIGINFO
	_SA_RESTART = C.SA_RESTART
	_SA_ONSTACK = C.SA_ONSTACK

	_PTHREAD_CREATE_DETACHED = C.PTHREAD_CREATE_DETACHED

	__SC_PAGE_SIZE        = C._SC_PAGE_SIZE
	__SC_NPROCESSORS_ONLN = C._SC_NPROCESSORS_ONLN

	_F_SETFL = C.F_SETFL
	_F_GETFD = C.F_GETFD
	_F_GETFL = C.F_GETFL
)

type sigset C.sigset_t
type siginfo C.siginfo_t
type timespec C.struct_timespec
type timestruc C.struct_timestruc_t
type timeval C.struct_timeval
type itimerval C.struct_itimerval

type stackt C.stack_t
type sigcontext C.struct_sigcontext
type ucontext C.ucontext_t
type _Ctype_struct___extctx uint64 // ucontext use a pointer to this structure but it shouldn't be used
type jmpbuf C.struct___jmpbuf
type context64 C.struct___context64
type sigactiont C.struct_sigaction
type tstate C.struct_tstate
type rusage C.struct_rusage

type pthread C.pthread_t
type pthread_attr C.pthread_attr_t

type semt C.sem_t
