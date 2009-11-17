// godefs -f -m64 freebsd/defs.c

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

typedef struct Sigaltstack Sigaltstack;
struct Sigaltstack {
	int8 *ss_sp;
	uint64 ss_size;
	int32 ss_flags;
	byte pad0[4];
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
	byte pad0[4];
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
	byte pad0[12];
};

typedef struct Sigcontext Sigcontext;
struct Sigcontext {
	Sigset sc_mask;
	int64 sc_onstack;
	int64 sc_rdi;
	int64 sc_rsi;
	int64 sc_rdx;
	int64 sc_rcx;
	int64 sc_r8;
	int64 sc_r9;
	int64 sc_rax;
	int64 sc_rbx;
	int64 sc_rbp;
	int64 sc_r10;
	int64 sc_r11;
	int64 sc_r12;
	int64 sc_r13;
	int64 sc_r14;
	int64 sc_r15;
	int32 sc_trapno;
	int16 sc_fs;
	int16 sc_gs;
	int64 sc_addr;
	int32 sc_flags;
	int16 sc_es;
	int16 sc_ds;
	int64 sc_err;
	int64 sc_rip;
	int64 sc_cs;
	int64 sc_rflags;
	int64 sc_rsp;
	int64 sc_ss;
	int64 sc_len;
	int64 sc_fpformat;
	int64 sc_ownedfp;
	int64 sc_fpstate[64];
	int64 sc_fsbase;
	int64 sc_gsbase;
	int64 sc_spare[6];
};
#pragma pack off
