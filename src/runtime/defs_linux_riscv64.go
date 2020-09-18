// Generated using cgo, then manually converted into appropriate naming and code
// for the Go runtime.
// go tool cgo -godefs defs_linux.go defs1_linux.go defs2_linux.go

package runtime

const (
	_EINTR  = 0x4
	_EAGAIN = 0xb
	_ENOMEM = 0xc
	_ENOSYS = 0x26

	_PROT_NONE  = 0x0
	_PROT_READ  = 0x1
	_PROT_WRITE = 0x2
	_PROT_EXEC  = 0x4

	_MAP_ANON    = 0x20
	_MAP_PRIVATE = 0x2
	_MAP_FIXED   = 0x10

	_MADV_DONTNEED   = 0x4
	_MADV_FREE       = 0x8
	_MADV_HUGEPAGE   = 0xe
	_MADV_NOHUGEPAGE = 0xf

	_SA_RESTART  = 0x10000000
	_SA_ONSTACK  = 0x8000000
	_SA_RESTORER = 0x0
	_SA_SIGINFO  = 0x4

	_SIGHUP    = 0x1
	_SIGINT    = 0x2
	_SIGQUIT   = 0x3
	_SIGILL    = 0x4
	_SIGTRAP   = 0x5
	_SIGABRT   = 0x6
	_SIGBUS    = 0x7
	_SIGFPE    = 0x8
	_SIGKILL   = 0x9
	_SIGUSR1   = 0xa
	_SIGSEGV   = 0xb
	_SIGUSR2   = 0xc
	_SIGPIPE   = 0xd
	_SIGALRM   = 0xe
	_SIGSTKFLT = 0x10
	_SIGCHLD   = 0x11
	_SIGCONT   = 0x12
	_SIGSTOP   = 0x13
	_SIGTSTP   = 0x14
	_SIGTTIN   = 0x15
	_SIGTTOU   = 0x16
	_SIGURG    = 0x17
	_SIGXCPU   = 0x18
	_SIGXFSZ   = 0x19
	_SIGVTALRM = 0x1a
	_SIGPROF   = 0x1b
	_SIGWINCH  = 0x1c
	_SIGIO     = 0x1d
	_SIGPWR    = 0x1e
	_SIGSYS    = 0x1f

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
	tv_sec  int64
	tv_nsec int64
}

//go:nosplit
func (ts *timespec) setNsec(ns int64) {
	ts.tv_sec = ns / 1e9
	ts.tv_nsec = ns % 1e9
}

type timeval struct {
	tv_sec  int64
	tv_usec int64
}

func (tv *timeval) set_usec(x int32) {
	tv.tv_usec = int64(x)
}

type sigactiont struct {
	sa_handler  uintptr
	sa_flags    uint64
	sa_restorer uintptr
	sa_mask     uint64
}

type siginfo struct {
	si_signo int32
	si_errno int32
	si_code  int32
	// below here is a union; si_addr is the only field we use
	si_addr uint64
}

type itimerval struct {
	it_interval timeval
	it_value    timeval
}

type epollevent struct {
	events    uint32
	pad_cgo_0 [4]byte
	data      [8]byte // unaligned uintptr
}

const (
	_O_RDONLY   = 0x0
	_O_NONBLOCK = 0x800
	_O_CLOEXEC  = 0x80000
)

type user_regs_struct struct {
	pc  uint64
	ra  uint64
	sp  uint64
	gp  uint64
	tp  uint64
	t0  uint64
	t1  uint64
	t2  uint64
	s0  uint64
	s1  uint64
	a0  uint64
	a1  uint64
	a2  uint64
	a3  uint64
	a4  uint64
	a5  uint64
	a6  uint64
	a7  uint64
	s2  uint64
	s3  uint64
	s4  uint64
	s5  uint64
	s6  uint64
	s7  uint64
	s8  uint64
	s9  uint64
	s10 uint64
	s11 uint64
	t3  uint64
	t4  uint64
	t5  uint64
	t6  uint64
}

type user_fpregs_struct struct {
	f [528]byte
}

type usigset struct {
	us_x__val [16]uint64
}

type sigcontext struct {
	sc_regs   user_regs_struct
	sc_fpregs user_fpregs_struct
}

type stackt struct {
	ss_sp    *byte
	ss_flags int32
	ss_size  uintptr
}

type ucontext struct {
	uc_flags     uint64
	uc_link      *ucontext
	uc_stack     stackt
	uc_sigmask   usigset
	uc_x__unused [0]uint8
	uc_pad_cgo_0 [8]byte
	uc_mcontext  sigcontext
}
