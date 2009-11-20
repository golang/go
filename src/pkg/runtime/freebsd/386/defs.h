// godefs freebsd/defs.c

// MACHINE GENERATED - DO NOT EDIT.

// Constants
enum {
	PROT_NONE = 0,
	PROT_READ = 0x1,
	PROT_WRITE = 0x2,
	PROT_EXEC = 0x4,
	MAP_ANON = 0x1000,
	MAP_PRIVATE = 0x2,
	SA_SIGINFO = 0x40,
	SA_RESTART = 0x2,
	SA_ONSTACK = 0x1,
	UMTX_OP_WAIT = 0x2,
	UMTX_OP_WAKE = 0x3,
	EINTR = 0x4,
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
	uint32 stack_size;
	int8 *tls_base;
	uint32 tls_size;
	int32 *child_tid;
	int32 *parent_tid;
	int32 flags;
	Rtprio *rtp;
	void* spare[3];
};

typedef struct Sigaltstack Sigaltstack;
struct Sigaltstack {
	int8 *ss_sp;
	uint32 ss_size;
	int32 ss_flags;
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
	uint32 ss_size;
	int32 ss_flags;
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
	byte _reason[32];
};

typedef struct Mcontext Mcontext;
struct Mcontext {
	int32 mc_onstack;
	int32 mc_gs;
	int32 mc_fs;
	int32 mc_es;
	int32 mc_ds;
	int32 mc_edi;
	int32 mc_esi;
	int32 mc_ebp;
	int32 mc_isp;
	int32 mc_ebx;
	int32 mc_edx;
	int32 mc_ecx;
	int32 mc_eax;
	int32 mc_trapno;
	int32 mc_err;
	int32 mc_eip;
	int32 mc_cs;
	int32 mc_eflags;
	int32 mc_esp;
	int32 mc_ss;
	int32 mc_len;
	int32 mc_fpformat;
	int32 mc_ownedfp;
	int32 mc_spare1[1];
	int32 mc_fpstate[128];
	int32 mc_fsbase;
	int32 mc_gsbase;
	int32 mc_spare2[6];
};

typedef struct Ucontext Ucontext;
struct Ucontext {
	Sigset uc_sigmask;
	Mcontext uc_mcontext;
	Ucontext *uc_link;
	StackT uc_stack;
	int32 uc_flags;
	int32 __spare__[4];
	byte pad0[12];
};
#pragma pack off
