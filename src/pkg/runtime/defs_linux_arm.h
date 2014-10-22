// TODO: Generate using cgo like defs_linux_{386,amd64}.h

// Constants
enum {
	EINTR  = 0x4,
	ENOMEM = 0xc,
	EAGAIN = 0xb,

	PROT_NONE = 0,
	PROT_READ = 0x1,
	PROT_WRITE = 0x2,
	PROT_EXEC = 0x4,
	MAP_ANON = 0x20,
	MAP_PRIVATE = 0x2,
	MAP_FIXED = 0x10,
	MADV_DONTNEED = 0x4,
	SA_RESTART = 0x10000000,
	SA_ONSTACK = 0x8000000,
	SA_RESTORER = 0, // unused on ARM
	SA_SIGINFO = 0x4,
	SIGHUP = 0x1,
	SIGINT = 0x2,
	SIGQUIT = 0x3,
	SIGILL = 0x4,
	SIGTRAP = 0x5,
	SIGABRT = 0x6,
	SIGBUS = 0x7,
	SIGFPE = 0x8,
	SIGKILL = 0x9,
	SIGUSR1 = 0xa,
	SIGSEGV = 0xb,
	SIGUSR2 = 0xc,
	SIGPIPE = 0xd,
	SIGALRM = 0xe,
	SIGSTKFLT = 0x10,
	SIGCHLD = 0x11,
	SIGCONT = 0x12,
	SIGSTOP = 0x13,
	SIGTSTP = 0x14,
	SIGTTIN = 0x15,
	SIGTTOU = 0x16,
	SIGURG = 0x17,
	SIGXCPU = 0x18,
	SIGXFSZ = 0x19,
	SIGVTALRM = 0x1a,
	SIGPROF = 0x1b,
	SIGWINCH = 0x1c,
	SIGIO = 0x1d,
	SIGPWR = 0x1e,
	SIGSYS = 0x1f,
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
	ITIMER_PROF = 0x2,
	ITIMER_VIRTUAL = 0x1,
	O_RDONLY = 0,
	O_CLOEXEC = 02000000,

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

// Types
#pragma pack on

typedef struct Timespec Timespec;
struct Timespec {
	int32 tv_sec;
	int32 tv_nsec;
};

typedef struct SigaltstackT SigaltstackT;
struct SigaltstackT {
	void *ss_sp;
	int32 ss_flags;
	uint32 ss_size;
};

typedef struct Sigcontext Sigcontext;
struct Sigcontext {
	uint32 trap_no;
	uint32 error_code;
	uint32 oldmask;
	uint32 arm_r0;
	uint32 arm_r1;
	uint32 arm_r2;
	uint32 arm_r3;
	uint32 arm_r4;
	uint32 arm_r5;
	uint32 arm_r6;
	uint32 arm_r7;
	uint32 arm_r8;
	uint32 arm_r9;
	uint32 arm_r10;
	uint32 arm_fp;
	uint32 arm_ip;
	uint32 arm_sp;
	uint32 arm_lr;
	uint32 arm_pc;
	uint32 arm_cpsr;
	uint32 fault_address;
};

typedef struct Ucontext Ucontext;
struct Ucontext {
	uint32 uc_flags;
	Ucontext *uc_link;
	SigaltstackT uc_stack;
	Sigcontext uc_mcontext;
	uint32 uc_sigmask;
	int32 __unused[31];
	uint32 uc_regspace[128];
};

typedef struct Timeval Timeval;
struct Timeval {
	int32 tv_sec;
	int32 tv_usec;
};

typedef struct Itimerval Itimerval;
struct Itimerval {
	Timeval it_interval;
	Timeval it_value;
};

typedef struct Siginfo Siginfo;
struct Siginfo {
	int32 si_signo;
	int32 si_errno;
	int32 si_code;
	uint8 _sifields[4];
};

typedef struct SigactionT SigactionT;
struct SigactionT {
	void *sa_handler;
	uint32 sa_flags;
	void *sa_restorer;
	uint64 sa_mask;
};

typedef struct EpollEvent EpollEvent;
struct EpollEvent {
	uint32	events;
	uint32	_pad;
	byte	data[8]; // to match amd64
};
#pragma pack off
