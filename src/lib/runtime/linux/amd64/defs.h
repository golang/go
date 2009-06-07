// godefs -f -m64 defs.c

// MACHINE GENERATED - DO NOT EDIT.

// Constants
enum {
	PROT_NONE = 0,
	PROT_READ = 0x1,
	PROT_WRITE = 0x2,
	PROT_EXEC = 0x4,
	MAP_ANON = 0x20,
	MAP_PRIVATE = 0x2,
	SA_RESTART = 0x10000000,
	SA_ONSTACK = 0x8000000,
	SA_RESTORER = 0x4000000,
	SA_SIGINFO = 0x4,
};

// Types
#pragma pack on

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

typedef struct Sigaction Sigaction;
struct Sigaction {
	void *sa_handler;
	uint64 sa_flags;
	void *sa_restorer;
	uint64 sa_mask;
};

typedef struct Siginfo Siginfo;
struct Siginfo {
	int32 si_signo;
	int32 si_errno;
	int32 si_code;
	byte pad0[4];
	byte _sifields[112];
};
#pragma pack off
// godefs -f -m64 defs1.c

// MACHINE GENERATED - DO NOT EDIT.

// Constants

// Types
#pragma pack on

typedef struct Usigset Usigset;
struct Usigset {
	uint64 __val[16];
};

typedef struct Fpxreg Fpxreg;
struct Fpxreg {
	uint16 significand[4];
	uint16 exponent;
	uint16 padding[3];
};

typedef struct Xmmreg Xmmreg;
struct Xmmreg {
	uint32 element[4];
};

typedef struct Fpstate Fpstate;
struct Fpstate {
	uint16 cwd;
	uint16 swd;
	uint16 ftw;
	uint16 fop;
	uint64 rip;
	uint64 rdp;
	uint32 mxcsr;
	uint32 mxcr_mask;
	Fpxreg _st[8];
	Xmmreg _xmm[16];
	uint32 padding[24];
};

typedef struct Fpxreg1 Fpxreg1;
struct Fpxreg1 {
	uint16 significand[4];
	uint16 exponent;
	uint16 padding[3];
};

typedef struct Xmmreg1 Xmmreg1;
struct Xmmreg1 {
	uint32 element[4];
};

typedef struct Fpstate1 Fpstate1;
struct Fpstate1 {
	uint16 cwd;
	uint16 swd;
	uint16 ftw;
	uint16 fop;
	uint64 rip;
	uint64 rdp;
	uint32 mxcsr;
	uint32 mxcr_mask;
	Fpxreg1 _st[8];
	Xmmreg1 _xmm[16];
	uint32 padding[24];
};

typedef struct Sigaltstack Sigaltstack;
struct Sigaltstack {
	void *ss_sp;
	int32 ss_flags;
	byte pad0[4];
	uint64 ss_size;
};

typedef struct Mcontext Mcontext;
struct Mcontext {
	int64 gregs[23];
	Fpstate *fpregs;
	uint64 __reserved1[8];
};

typedef struct Ucontext Ucontext;
struct Ucontext {
	uint64 uc_flags;
	Ucontext *uc_link;
	Sigaltstack uc_stack;
	Mcontext uc_mcontext;
	Usigset uc_sigmask;
	Fpstate __fpregs_mem;
};

typedef struct Sigcontext Sigcontext;
struct Sigcontext {
	uint64 r8;
	uint64 r9;
	uint64 r10;
	uint64 r11;
	uint64 r12;
	uint64 r13;
	uint64 r14;
	uint64 r15;
	uint64 rdi;
	uint64 rsi;
	uint64 rbp;
	uint64 rbx;
	uint64 rdx;
	uint64 rax;
	uint64 rcx;
	uint64 rsp;
	uint64 rip;
	uint64 eflags;
	uint16 cs;
	uint16 gs;
	uint16 fs;
	uint16 __pad0;
	uint64 err;
	uint64 trapno;
	uint64 oldmask;
	uint64 cr2;
	Fpstate1 *fpstate;
	uint64 __reserved1[8];
};
#pragma pack off
