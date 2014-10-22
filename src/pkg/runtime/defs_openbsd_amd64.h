// Created by cgo -cdefs - DO NOT EDIT
// cgo -cdefs defs_openbsd.go


enum {
	EINTR	= 0x4,
	EFAULT	= 0xe,

	PROT_NONE	= 0x0,
	PROT_READ	= 0x1,
	PROT_WRITE	= 0x2,
	PROT_EXEC	= 0x4,

	MAP_ANON	= 0x1000,
	MAP_PRIVATE	= 0x2,
	MAP_FIXED	= 0x10,

	MADV_FREE	= 0x6,

	SA_SIGINFO	= 0x40,
	SA_RESTART	= 0x2,
	SA_ONSTACK	= 0x1,

	SIGHUP		= 0x1,
	SIGINT		= 0x2,
	SIGQUIT		= 0x3,
	SIGILL		= 0x4,
	SIGTRAP		= 0x5,
	SIGABRT		= 0x6,
	SIGEMT		= 0x7,
	SIGFPE		= 0x8,
	SIGKILL		= 0x9,
	SIGBUS		= 0xa,
	SIGSEGV		= 0xb,
	SIGSYS		= 0xc,
	SIGPIPE		= 0xd,
	SIGALRM		= 0xe,
	SIGTERM		= 0xf,
	SIGURG		= 0x10,
	SIGSTOP		= 0x11,
	SIGTSTP		= 0x12,
	SIGCONT		= 0x13,
	SIGCHLD		= 0x14,
	SIGTTIN		= 0x15,
	SIGTTOU		= 0x16,
	SIGIO		= 0x17,
	SIGXCPU		= 0x18,
	SIGXFSZ		= 0x19,
	SIGVTALRM	= 0x1a,
	SIGPROF		= 0x1b,
	SIGWINCH	= 0x1c,
	SIGINFO		= 0x1d,
	SIGUSR1		= 0x1e,
	SIGUSR2		= 0x1f,

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

	EV_ADD		= 0x1,
	EV_DELETE	= 0x2,
	EV_CLEAR	= 0x20,
	EV_ERROR	= 0x4000,
	EVFILT_READ	= -0x1,
	EVFILT_WRITE	= -0x2,
};

typedef struct TforkT TforkT;
typedef struct SigaltstackT SigaltstackT;
typedef struct Sigcontext Sigcontext;
typedef struct Siginfo Siginfo;
typedef struct StackT StackT;
typedef struct Timespec Timespec;
typedef struct Timeval Timeval;
typedef struct Itimerval Itimerval;
typedef struct KeventT KeventT;

#pragma pack on

struct TforkT {
	byte	*tf_tcb;
	int32	*tf_tid;
	byte	*tf_stack;
};

struct SigaltstackT {
	byte	*ss_sp;
	uint64	ss_size;
	int32	ss_flags;
	byte	Pad_cgo_0[4];
};
struct Sigcontext {
	int64	sc_rdi;
	int64	sc_rsi;
	int64	sc_rdx;
	int64	sc_rcx;
	int64	sc_r8;
	int64	sc_r9;
	int64	sc_r10;
	int64	sc_r11;
	int64	sc_r12;
	int64	sc_r13;
	int64	sc_r14;
	int64	sc_r15;
	int64	sc_rbp;
	int64	sc_rbx;
	int64	sc_rax;
	int64	sc_gs;
	int64	sc_fs;
	int64	sc_es;
	int64	sc_ds;
	int64	sc_trapno;
	int64	sc_err;
	int64	sc_rip;
	int64	sc_cs;
	int64	sc_rflags;
	int64	sc_rsp;
	int64	sc_ss;
	void	*sc_fpstate;
	int32	__sc_unused;
	int32	sc_mask;
};
struct Siginfo {
	int32	si_signo;
	int32	si_code;
	int32	si_errno;
	byte	Pad_cgo_0[4];
	byte	_data[120];
};
typedef	uint32	Sigset;
typedef	byte	Sigval[8];

struct StackT {
	byte	*ss_sp;
	uint64	ss_size;
	int32	ss_flags;
	byte	Pad_cgo_0[4];
};

struct Timespec {
	int64	tv_sec;
	int64	tv_nsec;
};
struct Timeval {
	int64	tv_sec;
	int64	tv_usec;
};
struct Itimerval {
	Timeval	it_interval;
	Timeval	it_value;
};

struct KeventT {
	uint64	ident;
	int16	filter;
	uint16	flags;
	uint32	fflags;
	int64	data;
	byte	*udata;
};


#pragma pack off
