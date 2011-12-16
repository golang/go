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
	SA_SIGINFO = 0x40,
	SA_RESTART = 0x2,
	SA_ONSTACK = 0x1,
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
	FPE_INTDIV = 0x1,
	FPE_INTOVF = 0x2,
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

typedef struct Sigaltstack Sigaltstack;
struct Sigaltstack {
	void *ss_sp;
	uint64 ss_size;
	int32 ss_flags;
	byte pad_godefs_0[4];
};

typedef uint32 Sigset;

typedef struct Siginfo Siginfo;
struct Siginfo {
	int32 si_signo;
	int32 si_code;
	int32 si_errno;
	byte pad_godefs_0[4];
	byte _data[120];
};

typedef union Sigval Sigval;
union Sigval {
	int32 sival_int;
	void *sival_ptr;
};

typedef struct StackT StackT;
struct StackT {
	void *ss_sp;
	uint64 ss_size;
	int32 ss_flags;
	byte pad_godefs_0[4];
};

typedef struct Timespec Timespec;
struct Timespec {
	int32 tv_sec;
	byte pad_godefs_0[4];
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

typedef void sfxsave64;

typedef void usavefpu;

typedef struct Sigcontext Sigcontext;
struct Sigcontext {
	int64 sc_rdi;
	int64 sc_rsi;
	int64 sc_rdx;
	int64 sc_rcx;
	int64 sc_r8;
	int64 sc_r9;
	int64 sc_r10;
	int64 sc_r11;
	int64 sc_r12;
	int64 sc_r13;
	int64 sc_r14;
	int64 sc_r15;
	int64 sc_rbp;
	int64 sc_rbx;
	int64 sc_rax;
	int64 sc_gs;
	int64 sc_fs;
	int64 sc_es;
	int64 sc_ds;
	int64 sc_trapno;
	int64 sc_err;
	int64 sc_rip;
	int64 sc_cs;
	int64 sc_rflags;
	int64 sc_rsp;
	int64 sc_ss;
	sfxsave64 *sc_fpstate;
	int32 sc_onstack;
	int32 sc_mask;
};
#pragma pack off
