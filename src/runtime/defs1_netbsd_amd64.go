// created by cgo -cdefs and then converted to Go
// cgo -cdefs defs_netbsd.go defs_netbsd_amd64.go

package runtime

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

	_MADV_FREE = 0x6

	_SA_SIGINFO = 0x40
	_SA_RESTART = 0x2
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

	_EV_ADD       = 0x1
	_EV_DELETE    = 0x2
	_EV_CLEAR     = 0x20
	_EV_RECEIPT   = 0
	_EV_ERROR     = 0x4000
	_EVFILT_READ  = 0x0
	_EVFILT_WRITE = 0x1
)

type sigset struct {
	__bits [4]uint32
}

type siginfo struct {
	_signo  int32
	_code   int32
	_errno  int32
	_pad    int32
	_reason [24]byte
}

type stackt struct {
	ss_sp     uintptr
	ss_size   uintptr
	ss_flags  int32
	pad_cgo_0 [4]byte
}

type timespec struct {
	tv_sec  int64
	tv_nsec int64
}

func (ts *timespec) set_sec(x int32) {
	ts.tv_sec = int64(x)
}

func (ts *timespec) set_nsec(x int32) {
	ts.tv_nsec = int64(x)
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

type mcontextt struct {
	__gregs     [26]uint64
	_mc_tlsbase uint64
	__fpregs    [512]int8
}

type ucontextt struct {
	uc_flags    uint32
	pad_cgo_0   [4]byte
	uc_link     *ucontextt
	uc_sigmask  sigset
	uc_stack    stackt
	uc_mcontext mcontextt
}

type keventt struct {
	ident     uint64
	filter    uint32
	flags     uint32
	fflags    uint32
	pad_cgo_0 [4]byte
	data      int64
	udata     *byte
}

// created by cgo -cdefs and then converted to Go
// cgo -cdefs defs_netbsd.go defs_netbsd_amd64.go

const (
	_REG_RDI    = 0x0
	_REG_RSI    = 0x1
	_REG_RDX    = 0x2
	_REG_RCX    = 0x3
	_REG_R8     = 0x4
	_REG_R9     = 0x5
	_REG_R10    = 0x6
	_REG_R11    = 0x7
	_REG_R12    = 0x8
	_REG_R13    = 0x9
	_REG_R14    = 0xa
	_REG_R15    = 0xb
	_REG_RBP    = 0xc
	_REG_RBX    = 0xd
	_REG_RAX    = 0xe
	_REG_GS     = 0xf
	_REG_FS     = 0x10
	_REG_ES     = 0x11
	_REG_DS     = 0x12
	_REG_TRAPNO = 0x13
	_REG_ERR    = 0x14
	_REG_RIP    = 0x15
	_REG_CS     = 0x16
	_REG_RFLAGS = 0x17
	_REG_RSP    = 0x18
	_REG_SS     = 0x19
)
