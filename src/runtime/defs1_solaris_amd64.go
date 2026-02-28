// created by cgo -cdefs and then converted to Go
// cgo -cdefs defs_solaris.go defs_solaris_amd64.go

package runtime

const (
	_EINTR       = 0x4
	_EBADF       = 0x9
	_EFAULT      = 0xe
	_EAGAIN      = 0xb
	_EBUSY       = 0x10
	_ETIME       = 0x3e
	_ETIMEDOUT   = 0x91
	_EWOULDBLOCK = 0xb
	_EINPROGRESS = 0x96

	_PROT_NONE  = 0x0
	_PROT_READ  = 0x1
	_PROT_WRITE = 0x2
	_PROT_EXEC  = 0x4

	_MAP_ANON    = 0x100
	_MAP_PRIVATE = 0x2
	_MAP_FIXED   = 0x10

	_MADV_FREE = 0x5

	_SA_SIGINFO = 0x8
	_SA_RESTART = 0x4
	_SA_ONSTACK = 0x1

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
	_SIGURG    = 0x15
	_SIGSTOP   = 0x17
	_SIGTSTP   = 0x18
	_SIGCONT   = 0x19
	_SIGCHLD   = 0x12
	_SIGTTIN   = 0x1a
	_SIGTTOU   = 0x1b
	_SIGIO     = 0x16
	_SIGXCPU   = 0x1e
	_SIGXFSZ   = 0x1f
	_SIGVTALRM = 0x1c
	_SIGPROF   = 0x1d
	_SIGWINCH  = 0x14
	_SIGUSR1   = 0x10
	_SIGUSR2   = 0x11

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

	__SC_PAGESIZE         = 0xb
	__SC_NPROCESSORS_ONLN = 0xf

	_PTHREAD_CREATE_DETACHED = 0x40

	_FORK_NOSIGCHLD = 0x1
	_FORK_WAITPID   = 0x2

	_MAXHOSTNAMELEN = 0x100

	_O_NONBLOCK = 0x80
	_O_CLOEXEC  = 0x800000
	_FD_CLOEXEC = 0x1
	_F_GETFL    = 0x3
	_F_SETFL    = 0x4
	_F_SETFD    = 0x2

	_POLLIN  = 0x1
	_POLLOUT = 0x4
	_POLLHUP = 0x10
	_POLLERR = 0x8

	_PORT_SOURCE_FD    = 0x4
	_PORT_SOURCE_ALERT = 0x5
	_PORT_ALERT_UPDATE = 0x2
)

type semt struct {
	sem_count uint32
	sem_type  uint16
	sem_magic uint16
	sem_pad1  [3]uint64
	sem_pad2  [2]uint64
}

type sigset struct {
	__sigbits [4]uint32
}

type stackt struct {
	ss_sp     *byte
	ss_size   uintptr
	ss_flags  int32
	pad_cgo_0 [4]byte
}

type siginfo struct {
	si_signo int32
	si_code  int32
	si_errno int32
	si_pad   int32
	__data   [240]byte
}

type sigactiont struct {
	sa_flags  int32
	pad_cgo_0 [4]byte
	_funcptr  [8]byte
	sa_mask   sigset
}

type fpregset struct {
	fp_reg_set [528]byte
}

type mcontext struct {
	gregs  [28]int64
	fpregs fpregset
}

type ucontext struct {
	uc_flags    uint64
	uc_link     *ucontext
	uc_sigmask  sigset
	uc_stack    stackt
	pad_cgo_0   [8]byte
	uc_mcontext mcontext
	uc_filler   [5]int64
	pad_cgo_1   [8]byte
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

type portevent struct {
	portev_events int32
	portev_source uint16
	portev_pad    uint16
	portev_object uint64
	portev_user   *byte
}

type pthread uint32
type pthreadattr struct {
	__pthread_attrp *byte
}

type stat struct {
	st_dev     uint64
	st_ino     uint64
	st_mode    uint32
	st_nlink   uint32
	st_uid     uint32
	st_gid     uint32
	st_rdev    uint64
	st_size    int64
	st_atim    timespec
	st_mtim    timespec
	st_ctim    timespec
	st_blksize int32
	pad_cgo_0  [4]byte
	st_blocks  int64
	st_fstype  [16]int8
}

// created by cgo -cdefs and then converted to Go
// cgo -cdefs defs_solaris.go defs_solaris_amd64.go

const (
	_REG_RDI    = 0x8
	_REG_RSI    = 0x9
	_REG_RDX    = 0xc
	_REG_RCX    = 0xd
	_REG_R8     = 0x7
	_REG_R9     = 0x6
	_REG_R10    = 0x5
	_REG_R11    = 0x4
	_REG_R12    = 0x3
	_REG_R13    = 0x2
	_REG_R14    = 0x1
	_REG_R15    = 0x0
	_REG_RBP    = 0xa
	_REG_RBX    = 0xb
	_REG_RAX    = 0xe
	_REG_GS     = 0x17
	_REG_FS     = 0x16
	_REG_ES     = 0x18
	_REG_DS     = 0x19
	_REG_TRAPNO = 0xf
	_REG_ERR    = 0x10
	_REG_RIP    = 0x11
	_REG_CS     = 0x12
	_REG_RFLAGS = 0x13
	_REG_RSP    = 0x14
	_REG_SS     = 0x15
)
