// created by cgo -cdefs and then converted to Go
// cgo -cdefs defs_freebsd.go

package runtime

import "unsafe"

const (
	_NBBY            = 0x8
	_CTL_MAXNAME     = 0x18
	_CPU_LEVEL_WHICH = 0x3
	_CPU_WHICH_PID   = 0x2
)

const (
	_EINTR     = 0x4
	_EFAULT    = 0xe
	_EAGAIN    = 0x23
	_ETIMEDOUT = 0x3c

	_O_WRONLY   = 0x1
	_O_NONBLOCK = 0x4
	_O_CREAT    = 0x200
	_O_TRUNC    = 0x400
	_O_CLOEXEC  = 0x100000

	_PROT_NONE  = 0x0
	_PROT_READ  = 0x1
	_PROT_WRITE = 0x2
	_PROT_EXEC  = 0x4

	_MAP_ANON    = 0x1000
	_MAP_SHARED  = 0x1
	_MAP_PRIVATE = 0x2
	_MAP_FIXED   = 0x10

	_MADV_DONTNEED = 0x4
	_MADV_FREE     = 0x5

	_SA_SIGINFO = 0x40
	_SA_RESTART = 0x2
	_SA_ONSTACK = 0x1

	_CLOCK_MONOTONIC = 0x4
	_CLOCK_REALTIME  = 0x0

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
	_EV_EOF       = 0x8000
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

type thread int64 // long

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

type gpregs struct {
	gp_ra      uint64
	gp_sp      uint64
	gp_gp      uint64
	gp_tp      uint64
	gp_t       [7]uint64
	gp_s       [12]uint64
	gp_a       [8]uint64
	gp_sepc    uint64
	gp_sstatus uint64
}

type fpregs struct {
	fp_x     [64]uint64 // actually __uint64_t fp_x[32][2]
	fp_fcsr  uint64
	fp_flags int32
	pad      int32
}

type mcontext struct {
	mc_gpregs gpregs
	mc_fpregs fpregs
	mc_flags  int32
	mc_pad    int32
	mc_spare  [8]uint64
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
	ext    [4]uint64
}

type bintime struct {
	sec  int64
	frac uint64
}

type vdsoTimehands struct {
	algo         uint32
	gen          uint32
	scale        uint64
	offset_count uint32
	counter_mask uint32
	offset       bintime
	boottime     bintime
	physical     uint32
	res          [7]uint32
}

type vdsoTimekeep struct {
	ver       uint32
	enabled   uint32
	current   uint32
	pad_cgo_0 [4]byte
}

const (
	_VDSO_TK_VER_CURR = 0x1

	vdsoTimehandsSize = 0x58
	vdsoTimekeepSize  = 0x10
)
