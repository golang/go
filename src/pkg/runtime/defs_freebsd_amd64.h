// godefs -f -m64 defs.c

// MACHINE GENERATED - DO NOT EDIT.

// Constants
enum {
	PROT_NONE = 0,
	PROT_READ = 0x1,
	PROT_WRITE = 0x2,
	PROT_EXEC = 0x4,
	MAP_ANON = 0x1000,
	MAP_PRIVATE = 0x2,
	MAP_FIXED = 0x10,
	MADV_FREE = 0x5,
	SA_SIGINFO = 0x40,
	SA_RESTART = 0x2,
	SA_ONSTACK = 0x1,
	UMTX_OP_WAIT = 0x2,
	UMTX_OP_WAKE = 0x3,
	EINTR = 0x4,
	SIGHUP = 0x1,
	SIGINT = 0x2,
	SIGQUIT = 0x3,
	SIGILL = 0x4,
	SIGTRAP = 0x5,
	SIGABRT = 0x6,
	SIGEMT = 0x7,
	SIGFPE = 0x8,
	SIGKILL = 0x9,
	SIGBUS = 0xa,
	SIGSEGV = 0xb,
	SIGSYS = 0xc,
	SIGPIPE = 0xd,
	SIGALRM = 0xe,
	SIGTERM = 0xf,
	SIGURG = 0x10,
	SIGSTOP = 0x11,
	SIGTSTP = 0x12,
	SIGCONT = 0x13,
	SIGCHLD = 0x14,
	SIGTTIN = 0x15,
	SIGTTOU = 0x16,
	SIGIO = 0x17,
	SIGXCPU = 0x18,
	SIGXFSZ = 0x19,
	SIGVTALRM = 0x1a,
	SIGPROF = 0x1b,
	SIGWINCH = 0x1c,
	SIGINFO = 0x1d,
	SIGUSR1 = 0x1e,
	SIGUSR2 = 0x1f,
	FPE_INTDIV = 0x2,
	FPE_INTOVF = 0x1,
	FPE_FLTDIV = 0x3,
	FPE_FLTOVF = 0x4,
	FPE_FLTUND = 0x5,
	FPE_FLTRES = 0x6,
	FPE_FLTINV = 0x7,
	FPE_FLTSUB = 0x8,
	BUS_ADRALN = 0x1,
	BUS_ADRERR = 0x2,
	BUS_OBJERR = 0x3,
	SEGV_MAPERR = 0x1,
	SEGV_ACCERR = 0x2,
	ITIMER_REAL = 0,
	ITIMER_VIRTUAL = 0x1,
	ITIMER_PROF = 0x2,
};

// Types
#pragma pack on

typedef struct Rtprio Rtprio;
struct Rtprio {
	uint16 type;
	uint16 prio;
};

typedef struct ThrParam ThrParam;
struct ThrParam {
	void *start_func;
	void *arg;
	int8 *stack_base;
	uint64 stack_size;
	int8 *tls_base;
	uint64 tls_size;
	int64 *child_tid;
	int64 *parent_tid;
	int32 flags;
	byte pad_godefs_0[4];
	Rtprio *rtp;
	void* spare[3];
};

typedef struct Sigaltstack Sigaltstack;
struct Sigaltstack {
	int8 *ss_sp;
	uint64 ss_size;
	int32 ss_flags;
	byte pad_godefs_0[4];
};

typedef struct Sigset Sigset;
struct Sigset {
	uint32 __bits[4];
};

typedef union Sigval Sigval;
union Sigval {
	int32 sival_int;
	void *sival_ptr;
	int32 sigval_int;
	void *sigval_ptr;
};

typedef struct StackT StackT;
struct StackT {
	int8 *ss_sp;
	uint64 ss_size;
	int32 ss_flags;
	byte pad_godefs_0[4];
};

typedef struct Siginfo Siginfo;
struct Siginfo {
	int32 si_signo;
	int32 si_errno;
	int32 si_code;
	int32 si_pid;
	uint32 si_uid;
	int32 si_status;
	void *si_addr;
	Sigval si_value;
	byte _reason[40];
};

typedef struct Mcontext Mcontext;
struct Mcontext {
	int64 mc_onstack;
	int64 mc_rdi;
	int64 mc_rsi;
	int64 mc_rdx;
	int64 mc_rcx;
	int64 mc_r8;
	int64 mc_r9;
	int64 mc_rax;
	int64 mc_rbx;
	int64 mc_rbp;
	int64 mc_r10;
	int64 mc_r11;
	int64 mc_r12;
	int64 mc_r13;
	int64 mc_r14;
	int64 mc_r15;
	uint32 mc_trapno;
	uint16 mc_fs;
	uint16 mc_gs;
	int64 mc_addr;
	uint32 mc_flags;
	uint16 mc_es;
	uint16 mc_ds;
	int64 mc_err;
	int64 mc_rip;
	int64 mc_cs;
	int64 mc_rflags;
	int64 mc_rsp;
	int64 mc_ss;
	int64 mc_len;
	int64 mc_fpformat;
	int64 mc_ownedfp;
	int64 mc_fpstate[64];
	int64 mc_fsbase;
	int64 mc_gsbase;
	int64 mc_spare[6];
};

typedef struct Ucontext Ucontext;
struct Ucontext {
	Sigset uc_sigmask;
	Mcontext uc_mcontext;
	Ucontext *uc_link;
	StackT uc_stack;
	int32 uc_flags;
	int32 __spare__[4];
	byte pad_godefs_0[12];
};

typedef struct Timespec Timespec;
struct Timespec {
	int64 tv_sec;
	int64 tv_nsec;
};

typedef struct Timeval Timeval;
struct Timeval {
	int64 tv_sec;
	int64 tv_usec;
};

typedef struct Itimerval Itimerval;
struct Itimerval {
	Timeval it_interval;
	Timeval it_value;
};
#pragma pack off
