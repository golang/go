// Created by cgo -cdefs - DO NOT EDIT
// cgo -cdefs defs2_linux.go


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
	SA_RESTORER	= 0x4000000,
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

	O_RDONLY	= 0x0,
	O_CLOEXEC	= 0x80000,

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

typedef struct Fpreg Fpreg;
typedef struct Fpxreg Fpxreg;
typedef struct Xmmreg Xmmreg;
typedef struct Fpstate Fpstate;
typedef struct Timespec Timespec;
typedef struct Timeval Timeval;
typedef struct Sigaction Sigaction;
typedef struct Siginfo Siginfo;
typedef struct Sigaltstack Sigaltstack;
typedef struct Sigcontext Sigcontext;
typedef struct Ucontext Ucontext;
typedef struct Itimerval Itimerval;
typedef struct EpollEvent EpollEvent;

#pragma pack on

struct Fpreg {
	uint16	significand[4];
	uint16	exponent;
};
struct Fpxreg {
	uint16	significand[4];
	uint16	exponent;
	uint16	padding[3];
};
struct Xmmreg {
	uint32	element[4];
};
struct Fpstate {
	uint32	cw;
	uint32	sw;
	uint32	tag;
	uint32	ipoff;
	uint32	cssel;
	uint32	dataoff;
	uint32	datasel;
	Fpreg	_st[8];
	uint16	status;
	uint16	magic;
	uint32	_fxsr_env[6];
	uint32	mxcsr;
	uint32	reserved;
	Fpxreg	_fxsr_st[8];
	Xmmreg	_xmm[8];
	uint32	padding1[44];
	byte	anon0[48];
};
struct Timespec {
	int32	tv_sec;
	int32	tv_nsec;
};
struct Timeval {
	int32	tv_sec;
	int32	tv_usec;
};
struct Sigaction {
	void	*k_sa_handler;
	uint32	sa_flags;
	void	*sa_restorer;
	uint64	sa_mask;
};
struct Siginfo {
	int32	si_signo;
	int32	si_errno;
	int32	si_code;
	byte	_sifields[116];
};
struct Sigaltstack {
	byte	*ss_sp;
	int32	ss_flags;
	uint32	ss_size;
};
struct Sigcontext {
	uint16	gs;
	uint16	__gsh;
	uint16	fs;
	uint16	__fsh;
	uint16	es;
	uint16	__esh;
	uint16	ds;
	uint16	__dsh;
	uint32	edi;
	uint32	esi;
	uint32	ebp;
	uint32	esp;
	uint32	ebx;
	uint32	edx;
	uint32	ecx;
	uint32	eax;
	uint32	trapno;
	uint32	err;
	uint32	eip;
	uint16	cs;
	uint16	__csh;
	uint32	eflags;
	uint32	esp_at_signal;
	uint16	ss;
	uint16	__ssh;
	Fpstate	*fpstate;
	uint32	oldmask;
	uint32	cr2;
};
struct Ucontext {
	uint32	uc_flags;
	Ucontext	*uc_link;
	Sigaltstack	uc_stack;
	Sigcontext	uc_mcontext;
	uint32	uc_sigmask;
};
struct Itimerval {
	Timeval	it_interval;
	Timeval	it_value;
};
struct EpollEvent {
	uint32	events;
	uint64	data;
};


#pragma pack off
