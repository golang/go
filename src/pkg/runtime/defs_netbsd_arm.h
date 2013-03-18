// Created by cgo -cdefs - DO NOT EDIT
// cgo -cdefs defs_netbsd.go defs_netbsd_arm.go


enum {
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

	EINTR	= 0x4,

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
};

typedef struct Sigaltstack Sigaltstack;
typedef struct Sigset Sigset;
typedef struct Siginfo Siginfo;
typedef struct StackT StackT;
typedef struct Timespec Timespec;
typedef struct Timeval Timeval;
typedef struct Itimerval Itimerval;
typedef struct McontextT McontextT;
typedef struct UcontextT UcontextT;

#pragma pack on

struct Sigaltstack {
	byte	*ss_sp;
	uint32	ss_size;
	int32	ss_flags;
};
struct Sigset {
	uint32	__bits[4];
};
struct Siginfo {
	int32	_signo;
	int32	_code;
	int32	_errno;
	byte	_reason[20];
};

struct StackT {
	byte	*ss_sp;
	uint32	ss_size;
	int32	ss_flags;
};

struct Timespec {
	int64	tv_sec;
	int32	tv_nsec;
};
struct Timeval {
	int64	tv_sec;
	int32	tv_usec;
};
struct Itimerval {
	Timeval	it_interval;
	Timeval	it_value;
};

struct McontextT {
	uint32	__gregs[17];
#ifdef __ARM_EABI__
	byte	__fpu[4+8*32+4];
#else
	byte	__fpu[4+4*33+4];
#endif
	uint32	_mc_tlsbase;
};
struct UcontextT {
	uint32	uc_flags;
	UcontextT	*uc_link;
	Sigset	uc_sigmask;
	StackT	uc_stack;
	McontextT	uc_mcontext;
	int32	__uc_pad[2];
};

#pragma pack off
// Created by cgo -cdefs - DO NOT EDIT
// cgo -cdefs defs_netbsd.go defs_netbsd_arm.go


enum {
	REG_R0		= 0x0,
	REG_R1		= 0x1,
	REG_R2		= 0x2,
	REG_R3		= 0x3,
	REG_R4		= 0x4,
	REG_R5		= 0x5,
	REG_R6		= 0x6,
	REG_R7		= 0x7,
	REG_R8		= 0x8,
	REG_R9		= 0x9,
	REG_R10		= 0xa,
	REG_R11		= 0xb,
	REG_R12		= 0xc,
	REG_R13		= 0xd,
	REG_R14		= 0xe,
	REG_R15		= 0xf,
	REG_CPSR	= 0x10,
};

