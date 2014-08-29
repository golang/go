// Created by cgo -cdefs - DO NOT EDIT
// cgo -cdefs defs_dragonfly.go


enum {
	EINTR	= 0x4,
	EFAULT	= 0xe,
	EBUSY	= 0x10,
	EAGAIN	= 0x23,

	PROT_NONE	= 0x0,
	PROT_READ	= 0x1,
	PROT_WRITE	= 0x2,
	PROT_EXEC	= 0x4,

	MAP_ANON	= 0x1000,
	MAP_PRIVATE	= 0x2,
	MAP_FIXED	= 0x10,

	MADV_FREE	= 0x5,

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

	FPE_INTDIV	= 0x2,
	FPE_INTOVF	= 0x1,
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

typedef struct Rtprio Rtprio;
typedef struct Lwpparams Lwpparams;
typedef struct SigaltstackT SigaltstackT;
typedef struct Sigset Sigset;
typedef struct StackT StackT;
typedef struct Siginfo Siginfo;
typedef struct Mcontext Mcontext;
typedef struct Ucontext Ucontext;
typedef struct Timespec Timespec;
typedef struct Timeval Timeval;
typedef struct Itimerval Itimerval;
typedef struct KeventT KeventT;

#pragma pack on

struct Rtprio {
	uint16	type;
	uint16	prio;
};
struct Lwpparams {
	void	*func;
	byte	*arg;
	byte	*stack;
	int32	*tid1;
	int32	*tid2;
};
struct SigaltstackT {
	int8	*ss_sp;
	uint32	ss_size;
	int32	ss_flags;
};
struct Sigset {
	uint32	__bits[4];
};
struct StackT {
	int8	*ss_sp;
	uint32	ss_size;
	int32	ss_flags;
};

struct Siginfo {
	int32	si_signo;
	int32	si_errno;
	int32	si_code;
	int32	si_pid;
	uint32	si_uid;
	int32	si_status;
	byte	*si_addr;
	byte	si_value[4];
	int32	si_band;
	int32	__spare__[7];
};

struct Mcontext {
	int32	mc_onstack;
	int32	mc_gs;
	int32	mc_fs;
	int32	mc_es;
	int32	mc_ds;
	int32	mc_edi;
	int32	mc_esi;
	int32	mc_ebp;
	int32	mc_isp;
	int32	mc_ebx;
	int32	mc_edx;
	int32	mc_ecx;
	int32	mc_eax;
	int32	mc_xflags;
	int32	mc_trapno;
	int32	mc_err;
	int32	mc_eip;
	int32	mc_cs;
	int32	mc_eflags;
	int32	mc_esp;
	int32	mc_ss;
	int32	mc_len;
	int32	mc_fpformat;
	int32	mc_ownedfp;
	int32	mc_fpregs[128];
	int32	__spare__[16];
};
struct Ucontext {
	Sigset	uc_sigmask;
	Mcontext	uc_mcontext;
	Ucontext	*uc_link;
	StackT	uc_stack;
	int32	__spare__[8];
};

struct Timespec {
	int32	tv_sec;
	int32	tv_nsec;
};
struct Timeval {
	int32	tv_sec;
	int32	tv_usec;
};
struct Itimerval {
	Timeval	it_interval;
	Timeval	it_value;
};

struct KeventT {
	uint32	ident;
	int16	filter;
	uint16	flags;
	uint32	fflags;
	int32	data;
	byte	*udata;
};


#pragma pack off
