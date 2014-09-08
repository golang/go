// Created by cgo -cdefs - DO NOT EDIT
// cgo -cdefs defs_solaris.go defs_solaris_amd64.go


enum {
	EINTR		= 0x4,
	EBADF		= 0x9,
	EFAULT		= 0xe,
	EAGAIN		= 0xb,
	ETIMEDOUT	= 0x91,
	EWOULDBLOCK	= 0xb,
	EINPROGRESS	= 0x96,

	PROT_NONE	= 0x0,
	PROT_READ	= 0x1,
	PROT_WRITE	= 0x2,
	PROT_EXEC	= 0x4,

	MAP_ANON	= 0x100,
	MAP_PRIVATE	= 0x2,
	MAP_FIXED	= 0x10,

	MADV_FREE	= 0x5,

	SA_SIGINFO	= 0x8,
	SA_RESTART	= 0x4,
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
	SIGURG		= 0x15,
	SIGSTOP		= 0x17,
	SIGTSTP		= 0x18,
	SIGCONT		= 0x19,
	SIGCHLD		= 0x12,
	SIGTTIN		= 0x1a,
	SIGTTOU		= 0x1b,
	SIGIO		= 0x16,
	SIGXCPU		= 0x1e,
	SIGXFSZ		= 0x1f,
	SIGVTALRM	= 0x1c,
	SIGPROF		= 0x1d,
	SIGWINCH	= 0x14,
	SIGUSR1		= 0x10,
	SIGUSR2		= 0x11,

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

	_SC_NPROCESSORS_ONLN	= 0xf,

	PTHREAD_CREATE_DETACHED	= 0x40,

	FORK_NOSIGCHLD	= 0x1,
	FORK_WAITPID	= 0x2,

	MAXHOSTNAMELEN	= 0x100,

	O_NONBLOCK	= 0x80,
	FD_CLOEXEC	= 0x1,
	F_GETFL		= 0x3,
	F_SETFL		= 0x4,
	F_SETFD		= 0x2,

	POLLIN	= 0x1,
	POLLOUT	= 0x4,
	POLLHUP	= 0x10,
	POLLERR	= 0x8,

	PORT_SOURCE_FD	= 0x4,
};

typedef struct SemT SemT;
typedef struct SigaltstackT SigaltstackT;
typedef struct Sigset Sigset;
typedef struct StackT StackT;
typedef struct Siginfo Siginfo;
typedef struct SigactionT SigactionT;
typedef struct Fpregset Fpregset;
typedef struct Mcontext Mcontext;
typedef struct Ucontext Ucontext;
typedef struct Timespec Timespec;
typedef struct Timeval Timeval;
typedef struct Itimerval Itimerval;
typedef struct PortEvent PortEvent;
typedef struct PthreadAttr PthreadAttr;
typedef struct Stat Stat;

#pragma pack on

struct SemT {
	uint32	sem_count;
	uint16	sem_type;
	uint16	sem_magic;
	uint64	sem_pad1[3];
	uint64	sem_pad2[2];
};

struct SigaltstackT {
	byte	*ss_sp;
	uint64	ss_size;
	int32	ss_flags;
	byte	Pad_cgo_0[4];
};
struct Sigset {
	uint32	__sigbits[4];
};
struct StackT {
	byte	*ss_sp;
	uint64	ss_size;
	int32	ss_flags;
	byte	Pad_cgo_0[4];
};

struct Siginfo {
	int32	si_signo;
	int32	si_code;
	int32	si_errno;
	int32	si_pad;
	byte	__data[240];
};
struct SigactionT {
	int32	sa_flags;
	byte	Pad_cgo_0[4];
	byte	_funcptr[8];
	Sigset	sa_mask;
};

struct Fpregset {
	byte	fp_reg_set[528];
};
struct Mcontext {
	int64	gregs[28];
	Fpregset	fpregs;
};
struct Ucontext {
	uint64	uc_flags;
	Ucontext	*uc_link;
	Sigset	uc_sigmask;
	StackT	uc_stack;
	byte	Pad_cgo_0[8];
	Mcontext	uc_mcontext;
	int64	uc_filler[5];
	byte	Pad_cgo_1[8];
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

struct PortEvent {
	int32	portev_events;
	uint16	portev_source;
	uint16	portev_pad;
	uint64	portev_object;
	byte	*portev_user;
};
typedef	uint32	Pthread;
struct PthreadAttr {
	byte	*__pthread_attrp;
};

struct Stat {
	uint64	st_dev;
	uint64	st_ino;
	uint32	st_mode;
	uint32	st_nlink;
	uint32	st_uid;
	uint32	st_gid;
	uint64	st_rdev;
	int64	st_size;
	Timespec	st_atim;
	Timespec	st_mtim;
	Timespec	st_ctim;
	int32	st_blksize;
	byte	Pad_cgo_0[4];
	int64	st_blocks;
	int8	st_fstype[16];
};


#pragma pack off
// Created by cgo -cdefs - DO NOT EDIT
// cgo -cdefs defs_solaris.go defs_solaris_amd64.go


enum {
	REG_RDI		= 0x8,
	REG_RSI		= 0x9,
	REG_RDX		= 0xc,
	REG_RCX		= 0xd,
	REG_R8		= 0x7,
	REG_R9		= 0x6,
	REG_R10		= 0x5,
	REG_R11		= 0x4,
	REG_R12		= 0x3,
	REG_R13		= 0x2,
	REG_R14		= 0x1,
	REG_R15		= 0x0,
	REG_RBP		= 0xa,
	REG_RBX		= 0xb,
	REG_RAX		= 0xe,
	REG_GS		= 0x17,
	REG_FS		= 0x16,
	REG_ES		= 0x18,
	REG_DS		= 0x19,
	REG_TRAPNO	= 0xf,
	REG_ERR		= 0x10,
	REG_RIP		= 0x11,
	REG_CS		= 0x12,
	REG_RFLAGS	= 0x13,
	REG_RSP		= 0x14,
	REG_SS		= 0x15,
};

