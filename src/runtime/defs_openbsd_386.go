// created by cgo -cdefs and then converted to Go
// cgo -cdefs defs_openbsd.go

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
	_EV_ERROR     = 0x4000
	_EVFILT_READ  = -0x1
	_EVFILT_WRITE = -0x2
)

type tforkt struct {
	tf_tcb   *byte
	tf_tid   *int32
	tf_stack *byte
}

type sigaltstackt struct {
	ss_sp    *byte
	ss_size  uint32
	ss_flags int32
}

type sigcontext struct {
	sc_gs       int32
	sc_fs       int32
	sc_es       int32
	sc_ds       int32
	sc_edi      int32
	sc_esi      int32
	sc_ebp      int32
	sc_ebx      int32
	sc_edx      int32
	sc_ecx      int32
	sc_eax      int32
	sc_eip      int32
	sc_cs       int32
	sc_eflags   int32
	sc_esp      int32
	sc_ss       int32
	__sc_unused int32
	sc_mask     int32
	sc_trapno   int32
	sc_err      int32
	sc_fpstate  unsafe.Pointer
}

type siginfo struct {
	si_signo int32
	si_code  int32
	si_errno int32
	_data    [116]byte
}

type stackt struct {
	ss_sp    *byte
	ss_size  uint32
	ss_flags int32
}

type timespec struct {
	tv_sec  int64
	tv_nsec int32
}

type timeval struct {
	tv_sec  int64
	tv_usec int32
}

type itimerval struct {
	it_interval timeval
	it_value    timeval
}

type keventt struct {
	ident  uint32
	filter int16
	flags  uint16
	fflags uint32
	data   int64
	udata  *byte
}
