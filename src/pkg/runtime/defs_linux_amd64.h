// Created by cgo -cdefs - DO NOT EDIT
// cgo -cdefs defs_linux.go defs1_linux.go


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

typedef struct Timespec Timespec;
typedef struct Timeval Timeval;
typedef struct SigactionT SigactionT;
typedef struct Siginfo Siginfo;
typedef struct Itimerval Itimerval;
typedef struct EpollEvent EpollEvent;

#pragma pack on

struct Timespec {
	int64	tv_sec;
	int64	tv_nsec;
};
struct Timeval {
	int64	tv_sec;
	int64	tv_usec;
};
struct SigactionT {
	void	*sa_handler;
	uint64	sa_flags;
	void	*sa_restorer;
	uint64	sa_mask;
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
	byte	data[8]; // unaligned uintptr
};


#pragma pack off
// Created by cgo -cdefs - DO NOT EDIT
// cgo -cdefs defs_linux.go defs1_linux.go


enum {
	O_RDONLY	= 0x0,
	O_CLOEXEC	= 0x80000,
};

typedef struct Usigset Usigset;
typedef struct Fpxreg Fpxreg;
typedef struct Xmmreg Xmmreg;
typedef struct Fpstate Fpstate;
typedef struct Fpxreg1 Fpxreg1;
typedef struct Xmmreg1 Xmmreg1;
typedef struct Fpstate1 Fpstate1;
typedef struct Fpreg1 Fpreg1;
typedef struct SigaltstackT SigaltstackT;
typedef struct Mcontext Mcontext;
typedef struct Ucontext Ucontext;
typedef struct Sigcontext Sigcontext;

#pragma pack on

struct Usigset {
	uint64	__val[16];
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
	uint16	cwd;
	uint16	swd;
	uint16	ftw;
	uint16	fop;
	uint64	rip;
	uint64	rdp;
	uint32	mxcsr;
	uint32	mxcr_mask;
	Fpxreg	_st[8];
	Xmmreg	_xmm[16];
	uint32	padding[24];
};
struct Fpxreg1 {
	uint16	significand[4];
	uint16	exponent;
	uint16	padding[3];
};
struct Xmmreg1 {
	uint32	element[4];
};
struct Fpstate1 {
	uint16	cwd;
	uint16	swd;
	uint16	ftw;
	uint16	fop;
	uint64	rip;
	uint64	rdp;
	uint32	mxcsr;
	uint32	mxcr_mask;
	Fpxreg1	_st[8];
	Xmmreg1	_xmm[16];
	uint32	padding[24];
};
struct Fpreg1 {
	uint16	significand[4];
	uint16	exponent;
};
struct SigaltstackT {
	byte	*ss_sp;
	int32	ss_flags;
	byte	Pad_cgo_0[4];
	uint64	ss_size;
};
struct Mcontext {
	int64	gregs[23];
	Fpstate	*fpregs;
	uint64	__reserved1[8];
};
struct Ucontext {
	uint64	uc_flags;
	Ucontext	*uc_link;
	SigaltstackT	uc_stack;
	Mcontext	uc_mcontext;
	Usigset	uc_sigmask;
	Fpstate	__fpregs_mem;
};
struct Sigcontext {
	uint64	r8;
	uint64	r9;
	uint64	r10;
	uint64	r11;
	uint64	r12;
	uint64	r13;
	uint64	r14;
	uint64	r15;
	uint64	rdi;
	uint64	rsi;
	uint64	rbp;
	uint64	rbx;
	uint64	rdx;
	uint64	rax;
	uint64	rcx;
	uint64	rsp;
	uint64	rip;
	uint64	eflags;
	uint16	cs;
	uint16	gs;
	uint16	fs;
	uint16	__pad0;
	uint64	err;
	uint64	trapno;
	uint64	oldmask;
	uint64	cr2;
	Fpstate1	*fpstate;
	uint64	__reserved1[8];
};


#pragma pack off
