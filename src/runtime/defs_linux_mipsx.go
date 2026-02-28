// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build (mips || mipsle) && linux

package runtime

import "unsafe"

const (
	_EINTR  = 0x4
	_EAGAIN = 0xb
	_ENOMEM = 0xc

	_PROT_NONE  = 0x0
	_PROT_READ  = 0x1
	_PROT_WRITE = 0x2
	_PROT_EXEC  = 0x4

	_MAP_ANON    = 0x800
	_MAP_PRIVATE = 0x2
	_MAP_FIXED   = 0x10

	_MADV_DONTNEED   = 0x4
	_MADV_FREE       = 0x8
	_MADV_HUGEPAGE   = 0xe
	_MADV_NOHUGEPAGE = 0xf

	_SA_RESTART = 0x10000000
	_SA_ONSTACK = 0x8000000
	_SA_SIGINFO = 0x8

	_SI_KERNEL = 0x80
	_SI_TIMER  = -0x2

	_SIGHUP    = 0x1
	_SIGINT    = 0x2
	_SIGQUIT   = 0x3
	_SIGILL    = 0x4
	_SIGTRAP   = 0x5
	_SIGABRT   = 0x6
	_SIGEMT    = 0x7
	_SIGFPE    = 0x8
	_SIGKILL   = 0x9
	_SIGBUS    = 0xa
	_SIGSEGV   = 0xb
	_SIGSYS    = 0xc
	_SIGPIPE   = 0xd
	_SIGALRM   = 0xe
	_SIGUSR1   = 0x10
	_SIGUSR2   = 0x11
	_SIGCHLD   = 0x12
	_SIGPWR    = 0x13
	_SIGWINCH  = 0x14
	_SIGURG    = 0x15
	_SIGIO     = 0x16
	_SIGSTOP   = 0x17
	_SIGTSTP   = 0x18
	_SIGCONT   = 0x19
	_SIGTTIN   = 0x1a
	_SIGTTOU   = 0x1b
	_SIGVTALRM = 0x1c
	_SIGPROF   = 0x1d
	_SIGXCPU   = 0x1e
	_SIGXFSZ   = 0x1f

	_SIGRTMIN = 0x20

	_FPE_INTDIV = 0x1
	_FPE_INTOVF = 0x2
	_FPE_FLTDIV = 0x3
	_FPE_FLTOVF = 0x4
	_FPE_FLTUND = 0x5
	_FPE_FLTRES = 0x6
	_FPE_FLTINV = 0x7
	_FPE_FLTSUB = 0x8

	_BUS_ADRALN = 0x1
	_BUS_ADRERR = 0x2
	_BUS_OBJERR = 0x3

	_SEGV_MAPERR = 0x1
	_SEGV_ACCERR = 0x2

	_ITIMER_REAL    = 0x0
	_ITIMER_VIRTUAL = 0x1
	_ITIMER_PROF    = 0x2

	_CLOCK_THREAD_CPUTIME_ID = 0x3

	_SIGEV_THREAD_ID = 0x4

	_EPOLLIN       = 0x1
	_EPOLLOUT      = 0x4
	_EPOLLERR      = 0x8
	_EPOLLHUP      = 0x10
	_EPOLLRDHUP    = 0x2000
	_EPOLLET       = 0x80000000
	_EPOLL_CLOEXEC = 0x80000
	_EPOLL_CTL_ADD = 0x1
	_EPOLL_CTL_DEL = 0x2
	_EPOLL_CTL_MOD = 0x3
)

type timespec struct {
	tv_sec  int32
	tv_nsec int32
}

//go:nosplit
func (ts *timespec) setNsec(ns int64) {
	ts.tv_sec = timediv(ns, 1e9, &ts.tv_nsec)
}

type timeval struct {
	tv_sec  int32
	tv_usec int32
}

//go:nosplit
func (tv *timeval) set_usec(x int32) {
	tv.tv_usec = x
}

type sigactiont struct {
	sa_flags   uint32
	sa_handler uintptr
	sa_mask    [4]uint32
	// linux header does not have sa_restorer field,
	// but it is used in setsig(). it is no harm to put it here
	sa_restorer uintptr
}

type siginfoFields struct {
	si_signo int32
	si_code  int32
	si_errno int32
	// below here is a union; si_addr is the only field we use
	si_addr uint32
}

type siginfo struct {
	siginfoFields

	// Pad struct to the max size in the kernel.
	_ [_si_max_size - unsafe.Sizeof(siginfoFields{})]byte
}

type itimerspec struct {
	it_interval timespec
	it_value    timespec
}

type itimerval struct {
	it_interval timeval
	it_value    timeval
}

type sigeventFields struct {
	value  uintptr
	signo  int32
	notify int32
	// below here is a union; sigev_notify_thread_id is the only field we use
	sigev_notify_thread_id int32
}

type sigevent struct {
	sigeventFields

	// Pad struct to the max size in the kernel.
	_ [_sigev_max_size - unsafe.Sizeof(sigeventFields{})]byte
}

type epollevent struct {
	events    uint32
	pad_cgo_0 [4]byte
	data      uint64
}

const (
	_O_RDONLY    = 0x0
	_O_NONBLOCK  = 0x80
	_O_CLOEXEC   = 0x80000
	_SA_RESTORER = 0
)

type stackt struct {
	ss_sp    *byte
	ss_size  uintptr
	ss_flags int32
}

type sigcontext struct {
	sc_regmask   uint32
	sc_status    uint32
	sc_pc        uint64
	sc_regs      [32]uint64
	sc_fpregs    [32]uint64
	sc_acx       uint32
	sc_fpc_csr   uint32
	sc_fpc_eir   uint32
	sc_used_math uint32
	sc_dsp       uint32
	sc_mdhi      uint64
	sc_mdlo      uint64
	sc_hi1       uint32
	sc_lo1       uint32
	sc_hi2       uint32
	sc_lo2       uint32
	sc_hi3       uint32
	sc_lo3       uint32
}

type ucontext struct {
	uc_flags    uint32
	uc_link     *ucontext
	uc_stack    stackt
	Pad_cgo_0   [4]byte
	uc_mcontext sigcontext
	uc_sigmask  [4]uint32
}
