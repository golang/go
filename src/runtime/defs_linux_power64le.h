// Created by cgo -cdefs - DO NOT EDIT
// cgo -cdefs defs_linux.go defs3_linux.go


enum {
	EINTR	= 0x4,
	EAGAIN	= 0xb,
	ENOMEM	= 0xc,

	PROT_NONE	= 0x0,
	PROT_READ	= 0x1,
	PROT_WRITE	= 0x2,
	PROT_EXEC	= 0x4,

	MAP_ANON	= 0x20,
	MAP_PRIVATE	= 0x2,
	MAP_FIXED	= 0x10,

	MADV_DONTNEED	= 0x4,

	SA_RESTART	= 0x10000000,
	SA_ONSTACK	= 0x8000000,
	SA_SIGINFO	= 0x4,

	SIGHUP		= 0x1,
	SIGINT		= 0x2,
	SIGQUIT		= 0x3,
	SIGILL		= 0x4,
	SIGTRAP		= 0x5,
	SIGABRT		= 0x6,
	SIGBUS		= 0x7,
	SIGFPE		= 0x8,
	SIGKILL		= 0x9,
	SIGUSR1		= 0xa,
	SIGSEGV		= 0xb,
	SIGUSR2		= 0xc,
	SIGPIPE		= 0xd,
	SIGALRM		= 0xe,
	SIGSTKFLT	= 0x10,
	SIGCHLD		= 0x11,
	SIGCONT		= 0x12,
	SIGSTOP		= 0x13,
	SIGTSTP		= 0x14,
	SIGTTIN		= 0x15,
	SIGTTOU		= 0x16,
	SIGURG		= 0x17,
	SIGXCPU		= 0x18,
	SIGXFSZ		= 0x19,
	SIGVTALRM	= 0x1a,
	SIGPROF		= 0x1b,
	SIGWINCH	= 0x1c,
	SIGIO		= 0x1d,
	SIGPWR		= 0x1e,
	SIGSYS		= 0x1f,

	FPE_INTDIV	= 0x1,
	FPE_INTOVF	= 0x2,
	FPE_FLTDIV	= 0x3,
	FPE_FLTOVF	= 0x4,
	FPE_FLTUND	= 0x5,
	FPE_FLTRES	= 0x6,
	FPE_FLTINV	= 0x7,
	FPE_FLTSUB	= 0x8,

	BUS_ADRALN	= 0x1,
	BUS_ADRERR	= 0x2,
	BUS_OBJERR	= 0x3,

	SEGV_MAPERR	= 0x1,
	SEGV_ACCERR	= 0x2,

	ITIMER_REAL	= 0x0,
	ITIMER_VIRTUAL	= 0x1,
	ITIMER_PROF	= 0x2,

	EPOLLIN		= 0x1,
	EPOLLOUT	= 0x4,
	EPOLLERR	= 0x8,
	EPOLLHUP	= 0x10,
	EPOLLRDHUP	= 0x2000,
	EPOLLET		= -0x80000000,
	EPOLL_CLOEXEC	= 0x80000,
	EPOLL_CTL_ADD	= 0x1,
	EPOLL_CTL_DEL	= 0x2,
	EPOLL_CTL_MOD	= 0x3,
};

typedef struct Sigset Sigset;
typedef struct Timespec Timespec;
typedef struct Timeval Timeval;
typedef struct Sigaction Sigaction;
typedef struct Siginfo Siginfo;
typedef struct Itimerval Itimerval;
typedef struct EpollEvent EpollEvent;
typedef uint64 Usigset;

#pragma pack on

//struct Sigset {
//	uint64	sig[1];
//};
//typedef uint64 Sigset;

struct Timespec {
	int64	tv_sec;
	int64	tv_nsec;
};
struct Timeval {
	int64	tv_sec;
	int64	tv_usec;
};
struct Sigaction {
	void	*sa_handler;
	uint64	sa_flags;
	void	*sa_restorer;
	Usigset	sa_mask;
};
struct Siginfo {
	int32	si_signo;
	int32	si_errno;
	int32	si_code;
	byte	Pad_cgo_0[4];
	byte	_sifields[112];
};
struct Itimerval {
	Timeval	it_interval;
	Timeval	it_value;
};
struct EpollEvent {
	uint32	events;
	byte	Pad_cgo_0[4];
	uint64	data;
};


#pragma pack off
// Created by cgo -cdefs - DO NOT EDIT
// cgo -cdefs defs_linux.go defs3_linux.go


enum {
	O_RDONLY	= 0x0,
	O_CLOEXEC	= 0x80000,
	SA_RESTORER	= 0,
};

//typedef struct Usigset Usigset;
typedef struct Ptregs Ptregs;
typedef struct Vreg Vreg;
typedef struct Sigaltstack Sigaltstack;
typedef struct Sigcontext Sigcontext;
typedef struct Ucontext Ucontext;

#pragma pack on

//struct Usigset {
//	uint64	sig[1];
//};
//typedef Sigset Usigset;

struct Ptregs {
	uint64	gpr[32];
	uint64	nip;
	uint64	msr;
	uint64	orig_gpr3;
	uint64	ctr;
	uint64	link;
	uint64	xer;
	uint64	ccr;
	uint64	softe;
	uint64	trap;
	uint64	dar;
	uint64	dsisr;
	uint64	result;
};
typedef	uint64	Gregset[48];
typedef	float64	FPregset[33];
struct Vreg {
	uint32	u[4];
};

struct Sigaltstack {
	byte	*ss_sp;
	int32	ss_flags;
	byte	Pad_cgo_0[4];
	uint64	ss_size;
};

struct Sigcontext {
	uint64	_unused[4];
	int32	signal;
	int32	_pad0;
	uint64	handler;
	uint64	oldmask;
	Ptregs	*regs;
	uint64	gp_regs[48];
	float64	fp_regs[33];
	Vreg	*v_regs;
	int64	vmx_reserve[101];
};
struct Ucontext {
	uint64	uc_flags;
	Ucontext	*uc_link;
	Sigaltstack	uc_stack;
	Usigset	uc_sigmask;
	Usigset	__unused[15];
	Sigcontext	uc_mcontext;
};


#pragma pack off
