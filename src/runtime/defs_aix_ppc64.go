// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build aix

package runtime

const (
	_EPERM     = 0x1
	_ENOENT    = 0x2
	_EINTR     = 0x4
	_EAGAIN    = 0xb
	_ENOMEM    = 0xc
	_EACCES    = 0xd
	_EFAULT    = 0xe
	_EINVAL    = 0x16
	_ETIMEDOUT = 0x4e

	_PROT_NONE  = 0x0
	_PROT_READ  = 0x1
	_PROT_WRITE = 0x2
	_PROT_EXEC  = 0x4

	_MAP_ANONYMOUS = 0x10
	_MAP_PRIVATE   = 0x2
	_MAP_FIXED     = 0x100
	_MADV_DONTNEED = 0x4

	_SIGHUP     = 0x1
	_SIGINT     = 0x2
	_SIGQUIT    = 0x3
	_SIGILL     = 0x4
	_SIGTRAP    = 0x5
	_SIGABRT    = 0x6
	_SIGBUS     = 0xa
	_SIGFPE     = 0x8
	_SIGKILL    = 0x9
	_SIGUSR1    = 0x1e
	_SIGSEGV    = 0xb
	_SIGUSR2    = 0x1f
	_SIGPIPE    = 0xd
	_SIGALRM    = 0xe
	_SIGCHLD    = 0x14
	_SIGCONT    = 0x13
	_SIGSTOP    = 0x11
	_SIGTSTP    = 0x12
	_SIGTTIN    = 0x15
	_SIGTTOU    = 0x16
	_SIGURG     = 0x10
	_SIGXCPU    = 0x18
	_SIGXFSZ    = 0x19
	_SIGVTALRM  = 0x22
	_SIGPROF    = 0x20
	_SIGWINCH   = 0x1c
	_SIGIO      = 0x17
	_SIGPWR     = 0x1d
	_SIGSYS     = 0xc
	_SIGTERM    = 0xf
	_SIGEMT     = 0x7
	_SIGWAITING = 0x27

	_FPE_INTDIV = 0x14
	_FPE_INTOVF = 0x15
	_FPE_FLTDIV = 0x16
	_FPE_FLTOVF = 0x17
	_FPE_FLTUND = 0x18
	_FPE_FLTRES = 0x19
	_FPE_FLTINV = 0x1a
	_FPE_FLTSUB = 0x1b

	_BUS_ADRALN = 0x1
	_BUS_ADRERR = 0x2
	_BUS_OBJERR = 0x3
	_
	_SEGV_MAPERR = 0x32
	_SEGV_ACCERR = 0x33

	_ITIMER_REAL    = 0x0
	_ITIMER_VIRTUAL = 0x1
	_ITIMER_PROF    = 0x2

	_O_RDONLY = 0x0

	_SS_DISABLE  = 0x2
	_SI_USER     = 0x0
	_SIG_BLOCK   = 0x0
	_SIG_UNBLOCK = 0x1
	_SIG_SETMASK = 0x2

	_SA_SIGINFO = 0x100
	_SA_RESTART = 0x8
	_SA_ONSTACK = 0x1

	_PTHREAD_CREATE_DETACHED = 0x1

	__SC_PAGE_SIZE        = 0x30
	__SC_NPROCESSORS_ONLN = 0x48

	_F_SETFD    = 0x2
	_F_SETFL    = 0x4
	_F_GETFD    = 0x1
	_F_GETFL    = 0x3
	_FD_CLOEXEC = 0x1
)

type sigset [4]uint64

var sigset_all = sigset{^uint64(0), ^uint64(0), ^uint64(0), ^uint64(0)}

type siginfo struct {
	si_signo   int32
	si_errno   int32
	si_code    int32
	si_pid     int32
	si_uid     uint32
	si_status  int32
	si_addr    uintptr
	si_band    int64
	si_value   [2]int32 // [8]byte
	__si_flags int32
	__pad      [3]int32
}

type timespec struct {
	tv_sec  int64
	tv_nsec int64
}
type timeval struct {
	tv_sec    int64
	tv_usec   int32
	pad_cgo_0 [4]byte
}

func (tv *timeval) set_usec(x int32) {
	tv.tv_usec = x
}

type itimerval struct {
	it_interval timeval
	it_value    timeval
}

type stackt struct {
	ss_sp     uintptr
	ss_size   uintptr
	ss_flags  int32
	__pad     [4]int32
	pas_cgo_0 [4]byte
}

type sigcontext struct {
	sc_onstack int32
	pad_cgo_0  [4]byte
	sc_mask    sigset
	sc_uerror  int32
	sc_jmpbuf  context64
}

type ucontext struct {
	__sc_onstack   int32
	pad_cgo_0      [4]byte
	uc_sigmask     sigset
	__sc_error     int32
	pad_cgo_1      [4]byte
	uc_mcontext    context64
	uc_link        *ucontext
	uc_stack       stackt
	__extctx       uintptr // pointer to struct __extctx but we don't use it
	__extctx_magic int32
	__pad          int32
}

type context64 struct {
	gpr        [32]uint64
	msr        uint64
	iar        uint64
	lr         uint64
	ctr        uint64
	cr         uint32
	xer        uint32
	fpscr      uint32
	fpscrx     uint32
	except     [1]uint64
	fpr        [32]float64
	fpeu       uint8
	fpinfo     uint8
	fpscr24_31 uint8
	pad        [1]uint8
	excp_type  int32
}

type sigactiont struct {
	sa_handler uintptr // a union of two pointer
	sa_mask    sigset
	sa_flags   int32
	pad_cgo_0  [4]byte
}

type pthread uint32
type pthread_attr *byte

type semt int32
