// created by cgo -cdefs and then converted to Go
// cgo -cdefs defs2_linux.go

package runtime

const (
	_EINTR  = 0x4
	_EAGAIN = 0xb
	_ENOMEM = 0xc

	_PROT_NONE  = 0x0
	_PROT_READ  = 0x1
	_PROT_WRITE = 0x2
	_PROT_EXEC  = 0x4

	_MAP_ANON    = 0x20
	_MAP_PRIVATE = 0x2
	_MAP_FIXED   = 0x10

	_MADV_DONTNEED   = 0x4
	_MADV_HUGEPAGE   = 0xe
	_MADV_NOHUGEPAGE = 0xf

	_SA_RESTART  = 0x10000000
	_SA_ONSTACK  = 0x8000000
	_SA_RESTORER = 0x4000000
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

	_O_RDONLY  = 0x0
	_O_CLOEXEC = 0x80000

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

	_AF_UNIX    = 0x1
	_F_SETFL    = 0x4
	_SOCK_DGRAM = 0x2
)

type fpreg struct {
	significand [4]uint16
	exponent    uint16
}

type fpxreg struct {
	significand [4]uint16
	exponent    uint16
	padding     [3]uint16
}

type xmmreg struct {
	element [4]uint32
}

type fpstate struct {
	cw        uint32
	sw        uint32
	tag       uint32
	ipoff     uint32
	cssel     uint32
	dataoff   uint32
	datasel   uint32
	_st       [8]fpreg
	status    uint16
	magic     uint16
	_fxsr_env [6]uint32
	mxcsr     uint32
	reserved  uint32
	_fxsr_st  [8]fpxreg
	_xmm      [8]xmmreg
	padding1  [44]uint32
	anon0     [48]byte
}

type timespec struct {
	tv_sec  int32
	tv_nsec int32
}

func (ts *timespec) set_sec(x int64) {
	ts.tv_sec = int32(x)
}

func (ts *timespec) set_nsec(x int32) {
	ts.tv_nsec = x
}

type timeval struct {
	tv_sec  int32
	tv_usec int32
}

func (tv *timeval) set_usec(x int32) {
	tv.tv_usec = x
}

type sigactiont struct {
	sa_handler  uintptr
	sa_flags    uint32
	sa_restorer uintptr
	sa_mask     uint64
}

type siginfo struct {
	si_signo int32
	si_errno int32
	si_code  int32
	// below here is a union; si_addr is the only field we use
	si_addr uint32
}

type stackt struct {
	ss_sp    *byte
	ss_flags int32
	ss_size  uintptr
}

type sigcontext struct {
	gs            uint16
	__gsh         uint16
	fs            uint16
	__fsh         uint16
	es            uint16
	__esh         uint16
	ds            uint16
	__dsh         uint16
	edi           uint32
	esi           uint32
	ebp           uint32
	esp           uint32
	ebx           uint32
	edx           uint32
	ecx           uint32
	eax           uint32
	trapno        uint32
	err           uint32
	eip           uint32
	cs            uint16
	__csh         uint16
	eflags        uint32
	esp_at_signal uint32
	ss            uint16
	__ssh         uint16
	fpstate       *fpstate
	oldmask       uint32
	cr2           uint32
}

type ucontext struct {
	uc_flags    uint32
	uc_link     *ucontext
	uc_stack    stackt
	uc_mcontext sigcontext
	uc_sigmask  uint32
}

type itimerval struct {
	it_interval timeval
	it_value    timeval
}

type epollevent struct {
	events uint32
	data   [8]byte // to match amd64
}

type sockaddr_un struct {
	family uint16
	path   [108]byte
}
