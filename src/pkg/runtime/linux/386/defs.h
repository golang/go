// godefs -f -m32 -f -I/home/rsc/pub/linux-2.6/arch/x86/include -f -I/home/rsc/pub/linux-2.6/include defs2.c

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

typedef struct Fpreg Fpreg;
struct Fpreg {
	uint16 significand[4];
	uint16 exponent;
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
	uint32 cw;
	uint32 sw;
	uint32 tag;
	uint32 ipoff;
	uint32 cssel;
	uint32 dataoff;
	uint32 datasel;
	Fpreg _st[8];
	uint16 status;
	uint16 magic;
	uint32 _fxsr_env[6];
	uint32 mxcsr;
	uint32 reserved;
	Fpxreg _fxsr_st[8];
	Xmmreg _xmm[8];
	uint32 padding1[44];
	byte _anon_[48];
};

typedef struct Timespec Timespec;
struct Timespec {
	int32 tv_sec;
	int32 tv_nsec;
};

typedef struct Timeval Timeval;
struct Timeval {
	int32 tv_sec;
	int32 tv_usec;
};

typedef struct Sigaction Sigaction;
struct Sigaction {
	byte _u[4];
	uint32 sa_mask;
	uint32 sa_flags;
	void *sa_restorer;
};

typedef struct Siginfo Siginfo;
struct Siginfo {
	int32 si_signo;
	int32 si_errno;
	int32 si_code;
	byte _sifields[116];
};

typedef struct Sigaltstack Sigaltstack;
struct Sigaltstack {
	void *ss_sp;
	int32 ss_flags;
	uint32 ss_size;
};

typedef struct Sigcontext Sigcontext;
struct Sigcontext {
	uint16 gs;
	uint16 __gsh;
	uint16 fs;
	uint16 __fsh;
	uint16 es;
	uint16 __esh;
	uint16 ds;
	uint16 __dsh;
	uint32 edi;
	uint32 esi;
	uint32 ebp;
	uint32 esp;
	uint32 ebx;
	uint32 edx;
	uint32 ecx;
	uint32 eax;
	uint32 trapno;
	uint32 err;
	uint32 eip;
	uint16 cs;
	uint16 __csh;
	uint32 eflags;
	uint32 esp_at_signal;
	uint16 ss;
	uint16 __ssh;
	Fpstate *fpstate;
	uint32 oldmask;
	uint32 cr2;
};

typedef struct Ucontext Ucontext;
struct Ucontext {
	uint32 uc_flags;
	Ucontext *uc_link;
	Sigaltstack uc_stack;
	Sigcontext uc_mcontext;
	uint32 uc_sigmask;
};
#pragma pack off
