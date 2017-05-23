// created by cgo -cdefs and then converted to Go
// cgo -cdefs defs_freebsd.go

package runtime

import "unsafe"

const (
	_EINTR  = 0x4
	_EFAULT = 0xe

	_PROT_NONE  = 0x0
	_PROT_READ  = 0x1
	_PROT_WRITE = 0x2
	_PROT_EXEC  = 0x4

	_MAP_ANON    = 0x1000
	_MAP_PRIVATE = 0x2
	_MAP_FIXED   = 0x10

	_MADV_FREE = 0x5

	_SA_SIGINFO = 0x40
	_SA_RESTART = 0x2
	_SA_ONSTACK = 0x1

	_CLOCK_MONOTONIC = 0x4

	_UMTX_OP_WAIT_UINT         = 0xb
	_UMTX_OP_WAIT_UINT_PRIVATE = 0xf
	_UMTX_OP_WAKE              = 0x3
	_UMTX_OP_WAKE_PRIVATE      = 0x10

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
	_SIGTERM   = 0xf
	_SIGURG    = 0x10
	_SIGSTOP   = 0x11
	_SIGTSTP   = 0x12
	_SIGCONT   = 0x13
	_SIGCHLD   = 0x14
	_SIGTTIN   = 0x15
	_SIGTTOU   = 0x16
	_SIGIO     = 0x17
	_SIGXCPU   = 0x18
	_SIGXFSZ   = 0x19
	_SIGVTALRM = 0x1a
	_SIGPROF   = 0x1b
	_SIGWINCH  = 0x1c
	_SIGINFO   = 0x1d
	_SIGUSR1   = 0x1e
	_SIGUSR2   = 0x1f

	_FPE_INTDIV = 0x2
	_FPE_INTOVF = 0x1
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

	_EV_ADD       = 0x1
	_EV_DELETE    = 0x2
	_EV_CLEAR     = 0x20
	_EV_RECEIPT   = 0x40
	_EV_ERROR     = 0x4000
	_EVFILT_READ  = -0x1
	_EVFILT_WRITE = -0x2
)

type rtprio struct {
	_type uint16
	prio  uint16
}

type thrparam struct {
	start_func uintptr
	arg        unsafe.Pointer
	stack_base uintptr
	stack_size uintptr
	tls_base   unsafe.Pointer
	tls_size   uintptr
	child_tid  unsafe.Pointer // *int64
	parent_tid *int64
	flags      int32
	pad_cgo_0  [4]byte
	rtp        *rtprio
	spare      [3]uintptr
}

type sigaltstackt struct {
	ss_sp     *int8
	ss_size   uint64
	ss_flags  int32
	pad_cgo_0 [4]byte
}

type sigset struct {
	__bits [4]uint32
}

type stackt struct {
	ss_sp     uintptr
	ss_size   uintptr
	ss_flags  int32
	pad_cgo_0 [4]byte
}

type siginfo struct {
	si_signo  int32
	si_errno  int32
	si_code   int32
	si_pid    int32
	si_uid    uint32
	si_status int32
	si_addr   uint64
	si_value  [8]byte
	_reason   [40]byte
}

type mcontext struct {
	mc_onstack       uint64
	mc_rdi           uint64
	mc_rsi           uint64
	mc_rdx           uint64
	mc_rcx           uint64
	mc_r8            uint64
	mc_r9            uint64
	mc_rax           uint64
	mc_rbx           uint64
	mc_rbp           uint64
	mc_r10           uint64
	mc_r11           uint64
	mc_r12           uint64
	mc_r13           uint64
	mc_r14           uint64
	mc_r15           uint64
	mc_trapno        uint32
	mc_fs            uint16
	mc_gs            uint16
	mc_addr          uint64
	mc_flags         uint32
	mc_es            uint16
	mc_ds            uint16
	mc_err           uint64
	mc_rip           uint64
	mc_cs            uint64
	mc_rflags        uint64
	mc_rsp           uint64
	mc_ss            uint64
	mc_len           uint64
	mc_fpformat      uint64
	mc_ownedfp       uint64
	mc_fpstate       [64]uint64
	mc_fsbase        uint64
	mc_gsbase        uint64
	mc_xfpustate     uint64
	mc_xfpustate_len uint64
	mc_spare         [4]uint64
}

type ucontext struct {
	uc_sigmask  sigset
	uc_mcontext mcontext
	uc_link     *ucontext
	uc_stack    stackt
	uc_flags    int32
	__spare__   [4]int32
	pad_cgo_0   [12]byte
}

type timespec struct {
	tv_sec  int64
	tv_nsec int64
}

func (ts *timespec) set_sec(x int64) {
	ts.tv_sec = x
}

type timeval struct {
	tv_sec  int64
	tv_usec int64
}

func (tv *timeval) set_usec(x int32) {
	tv.tv_usec = int64(x)
}

type itimerval struct {
	it_interval timeval
	it_value    timeval
}

type umtx_time struct {
	_timeout timespec
	_flags   uint32
	_clockid uint32
}

type keventt struct {
	ident  uint64
	filter int16
	flags  uint16
	fflags uint32
	data   int64
	udata  *byte
}
