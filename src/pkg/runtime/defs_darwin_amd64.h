// Created by cgo -cdefs - DO NOT EDIT
// cgo -cdefs defs_darwin.go


enum {
	EINTR	= 0x4,
	EFAULT	= 0xe,

	PROT_NONE	= 0x0,
	PROT_READ	= 0x1,
	PROT_WRITE	= 0x2,
	PROT_EXEC	= 0x4,

	MAP_ANON	= 0x1000,
	MAP_PRIVATE	= 0x2,
	MAP_FIXED	= 0x10,

	MADV_DONTNEED	= 0x4,
	MADV_FREE	= 0x5,

	MACH_MSG_TYPE_MOVE_RECEIVE	= 0x10,
	MACH_MSG_TYPE_MOVE_SEND		= 0x11,
	MACH_MSG_TYPE_MOVE_SEND_ONCE	= 0x12,
	MACH_MSG_TYPE_COPY_SEND		= 0x13,
	MACH_MSG_TYPE_MAKE_SEND		= 0x14,
	MACH_MSG_TYPE_MAKE_SEND_ONCE	= 0x15,
	MACH_MSG_TYPE_COPY_RECEIVE	= 0x16,

	MACH_MSG_PORT_DESCRIPTOR		= 0x0,
	MACH_MSG_OOL_DESCRIPTOR			= 0x1,
	MACH_MSG_OOL_PORTS_DESCRIPTOR		= 0x2,
	MACH_MSG_OOL_VOLATILE_DESCRIPTOR	= 0x3,

	MACH_MSGH_BITS_COMPLEX	= 0x80000000,

	MACH_SEND_MSG	= 0x1,
	MACH_RCV_MSG	= 0x2,
	MACH_RCV_LARGE	= 0x4,

	MACH_SEND_TIMEOUT	= 0x10,
	MACH_SEND_INTERRUPT	= 0x40,
	MACH_SEND_ALWAYS	= 0x10000,
	MACH_SEND_TRAILER	= 0x20000,
	MACH_RCV_TIMEOUT	= 0x100,
	MACH_RCV_NOTIFY		= 0x200,
	MACH_RCV_INTERRUPT	= 0x400,
	MACH_RCV_OVERWRITE	= 0x1000,

	NDR_PROTOCOL_2_0	= 0x0,
	NDR_INT_BIG_ENDIAN	= 0x0,
	NDR_INT_LITTLE_ENDIAN	= 0x1,
	NDR_FLOAT_IEEE		= 0x0,
	NDR_CHAR_ASCII		= 0x0,

	SA_SIGINFO	= 0x40,
	SA_RESTART	= 0x2,
	SA_ONSTACK	= 0x1,
	SA_USERTRAMP	= 0x100,
	SA_64REGSET	= 0x200,

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

	FPE_INTDIV	= 0x7,
	FPE_INTOVF	= 0x8,
	FPE_FLTDIV	= 0x1,
	FPE_FLTOVF	= 0x2,
	FPE_FLTUND	= 0x3,
	FPE_FLTRES	= 0x4,
	FPE_FLTINV	= 0x5,
	FPE_FLTSUB	= 0x6,

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
	EV_RECEIPT	= 0x40,
	EV_ERROR	= 0x4000,
	EVFILT_READ	= -0x1,
	EVFILT_WRITE	= -0x2,
};

typedef struct MachBody MachBody;
typedef struct MachHeader MachHeader;
typedef struct MachNDR MachNDR;
typedef struct MachPort MachPort;
typedef struct StackT StackT;
typedef struct Sigaction Sigaction;
typedef struct Siginfo Siginfo;
typedef struct Timeval Timeval;
typedef struct Itimerval Itimerval;
typedef struct Timespec Timespec;
typedef struct FPControl FPControl;
typedef struct FPStatus FPStatus;
typedef struct RegMMST RegMMST;
typedef struct RegXMM RegXMM;
typedef struct Regs64 Regs64;
typedef struct FloatState64 FloatState64;
typedef struct ExceptionState64 ExceptionState64;
typedef struct Mcontext64 Mcontext64;
typedef struct Regs32 Regs32;
typedef struct FloatState32 FloatState32;
typedef struct ExceptionState32 ExceptionState32;
typedef struct Mcontext32 Mcontext32;
typedef struct Ucontext Ucontext;
typedef struct Kevent Kevent;

#pragma pack on

struct MachBody {
	uint32	msgh_descriptor_count;
};
struct MachHeader {
	uint32	msgh_bits;
	uint32	msgh_size;
	uint32	msgh_remote_port;
	uint32	msgh_local_port;
	uint32	msgh_reserved;
	int32	msgh_id;
};
struct MachNDR {
	uint8	mig_vers;
	uint8	if_vers;
	uint8	reserved1;
	uint8	mig_encoding;
	uint8	int_rep;
	uint8	char_rep;
	uint8	float_rep;
	uint8	reserved2;
};
struct MachPort {
	uint32	name;
	uint32	pad1;
	uint16	pad2;
	uint8	disposition;
	uint8	type;
};

struct StackT {
	byte	*ss_sp;
	uint64	ss_size;
	int32	ss_flags;
	byte	Pad_cgo_0[4];
};
typedef	byte	Sighandler[8];

struct Sigaction {
	byte	__sigaction_u[8];
	void	*sa_tramp;
	uint32	sa_mask;
	int32	sa_flags;
};

typedef	byte	Sigval[8];
struct Siginfo {
	int32	si_signo;
	int32	si_errno;
	int32	si_code;
	int32	si_pid;
	uint32	si_uid;
	int32	si_status;
	byte	*si_addr;
	byte	si_value[8];
	int64	si_band;
	uint64	__pad[7];
};
struct Timeval {
	int64	tv_sec;
	int32	tv_usec;
	byte	Pad_cgo_0[4];
};
struct Itimerval {
	Timeval	it_interval;
	Timeval	it_value;
};
struct Timespec {
	int64	tv_sec;
	int64	tv_nsec;
};

struct FPControl {
	byte	Pad_cgo_0[2];
};
struct FPStatus {
	byte	Pad_cgo_0[2];
};
struct RegMMST {
	int8	mmst_reg[10];
	int8	mmst_rsrv[6];
};
struct RegXMM {
	int8	xmm_reg[16];
};

struct Regs64 {
	uint64	rax;
	uint64	rbx;
	uint64	rcx;
	uint64	rdx;
	uint64	rdi;
	uint64	rsi;
	uint64	rbp;
	uint64	rsp;
	uint64	r8;
	uint64	r9;
	uint64	r10;
	uint64	r11;
	uint64	r12;
	uint64	r13;
	uint64	r14;
	uint64	r15;
	uint64	rip;
	uint64	rflags;
	uint64	cs;
	uint64	fs;
	uint64	gs;
};
struct FloatState64 {
	int32	fpu_reserved[2];
	FPControl	fpu_fcw;
	FPStatus	fpu_fsw;
	uint8	fpu_ftw;
	uint8	fpu_rsrv1;
	uint16	fpu_fop;
	uint32	fpu_ip;
	uint16	fpu_cs;
	uint16	fpu_rsrv2;
	uint32	fpu_dp;
	uint16	fpu_ds;
	uint16	fpu_rsrv3;
	uint32	fpu_mxcsr;
	uint32	fpu_mxcsrmask;
	RegMMST	fpu_stmm0;
	RegMMST	fpu_stmm1;
	RegMMST	fpu_stmm2;
	RegMMST	fpu_stmm3;
	RegMMST	fpu_stmm4;
	RegMMST	fpu_stmm5;
	RegMMST	fpu_stmm6;
	RegMMST	fpu_stmm7;
	RegXMM	fpu_xmm0;
	RegXMM	fpu_xmm1;
	RegXMM	fpu_xmm2;
	RegXMM	fpu_xmm3;
	RegXMM	fpu_xmm4;
	RegXMM	fpu_xmm5;
	RegXMM	fpu_xmm6;
	RegXMM	fpu_xmm7;
	RegXMM	fpu_xmm8;
	RegXMM	fpu_xmm9;
	RegXMM	fpu_xmm10;
	RegXMM	fpu_xmm11;
	RegXMM	fpu_xmm12;
	RegXMM	fpu_xmm13;
	RegXMM	fpu_xmm14;
	RegXMM	fpu_xmm15;
	int8	fpu_rsrv4[96];
	int32	fpu_reserved1;
};
struct ExceptionState64 {
	uint16	trapno;
	uint16	cpu;
	uint32	err;
	uint64	faultvaddr;
};
struct Mcontext64 {
	ExceptionState64	es;
	Regs64	ss;
	FloatState64	fs;
	byte	Pad_cgo_0[4];
};

struct Regs32 {
	uint32	eax;
	uint32	ebx;
	uint32	ecx;
	uint32	edx;
	uint32	edi;
	uint32	esi;
	uint32	ebp;
	uint32	esp;
	uint32	ss;
	uint32	eflags;
	uint32	eip;
	uint32	cs;
	uint32	ds;
	uint32	es;
	uint32	fs;
	uint32	gs;
};
struct FloatState32 {
	int32	fpu_reserved[2];
	FPControl	fpu_fcw;
	FPStatus	fpu_fsw;
	uint8	fpu_ftw;
	uint8	fpu_rsrv1;
	uint16	fpu_fop;
	uint32	fpu_ip;
	uint16	fpu_cs;
	uint16	fpu_rsrv2;
	uint32	fpu_dp;
	uint16	fpu_ds;
	uint16	fpu_rsrv3;
	uint32	fpu_mxcsr;
	uint32	fpu_mxcsrmask;
	RegMMST	fpu_stmm0;
	RegMMST	fpu_stmm1;
	RegMMST	fpu_stmm2;
	RegMMST	fpu_stmm3;
	RegMMST	fpu_stmm4;
	RegMMST	fpu_stmm5;
	RegMMST	fpu_stmm6;
	RegMMST	fpu_stmm7;
	RegXMM	fpu_xmm0;
	RegXMM	fpu_xmm1;
	RegXMM	fpu_xmm2;
	RegXMM	fpu_xmm3;
	RegXMM	fpu_xmm4;
	RegXMM	fpu_xmm5;
	RegXMM	fpu_xmm6;
	RegXMM	fpu_xmm7;
	int8	fpu_rsrv4[224];
	int32	fpu_reserved1;
};
struct ExceptionState32 {
	uint16	trapno;
	uint16	cpu;
	uint32	err;
	uint32	faultvaddr;
};
struct Mcontext32 {
	ExceptionState32	es;
	Regs32	ss;
	FloatState32	fs;
};

struct Ucontext {
	int32	uc_onstack;
	uint32	uc_sigmask;
	StackT	uc_stack;
	Ucontext	*uc_link;
	uint64	uc_mcsize;
	Mcontext64	*uc_mcontext;
};

struct Kevent {
	uint64	ident;
	int16	filter;
	uint16	flags;
	uint32	fflags;
	int64	data;
	byte	*udata;
};


#pragma pack off
