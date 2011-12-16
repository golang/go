// c:\go\bin\godefs.exe -f -m64 defs.c

// MACHINE GENERATED - DO NOT EDIT.

// Constants
enum {
	PROT_NONE = 0,
	PROT_READ = 0x1,
	PROT_WRITE = 0x2,
	PROT_EXEC = 0x4,
	MAP_ANON = 0x1,
	MAP_PRIVATE = 0x2,
	DUPLICATE_SAME_ACCESS = 0x2,
	THREAD_PRIORITY_HIGHEST = 0x2,
	SIGINT = 0x2,
	CTRL_C_EVENT = 0,
	CTRL_BREAK_EVENT = 0x1,
	CONTEXT_CONTROL = 0x100001,
	CONTEXT_FULL = 0x10000b,
	EXCEPTION_ACCESS_VIOLATION = 0xc0000005,
	EXCEPTION_BREAKPOINT = 0x80000003,
	EXCEPTION_FLT_DENORMAL_OPERAND = 0xc000008d,
	EXCEPTION_FLT_DIVIDE_BY_ZERO = 0xc000008e,
	EXCEPTION_FLT_INEXACT_RESULT = 0xc000008f,
	EXCEPTION_FLT_OVERFLOW = 0xc0000091,
	EXCEPTION_FLT_UNDERFLOW = 0xc0000093,
	EXCEPTION_INT_DIVIDE_BY_ZERO = 0xc0000094,
	EXCEPTION_INT_OVERFLOW = 0xc0000095,
};

// Types
#pragma pack on

typedef struct SystemInfo SystemInfo;
struct SystemInfo {
	byte Pad_godefs_0[4];
	uint32 dwPageSize;
	void *lpMinimumApplicationAddress;
	void *lpMaximumApplicationAddress;
	uint64 dwActiveProcessorMask;
	uint32 dwNumberOfProcessors;
	uint32 dwProcessorType;
	uint32 dwAllocationGranularity;
	uint16 wProcessorLevel;
	uint16 wProcessorRevision;
};

typedef struct ExceptionRecord ExceptionRecord;
struct ExceptionRecord {
	uint32 ExceptionCode;
	uint32 ExceptionFlags;
	ExceptionRecord *ExceptionRecord;
	void *ExceptionAddress;
	uint32 NumberParameters;
	byte pad_godefs_0[4];
	uint64 ExceptionInformation[15];
};

typedef struct M128a M128a;
struct M128a {
	uint64 Low;
	int64 High;
};

typedef struct Context Context;
struct Context {
	uint64 P1Home;
	uint64 P2Home;
	uint64 P3Home;
	uint64 P4Home;
	uint64 P5Home;
	uint64 P6Home;
	uint32 ContextFlags;
	uint32 MxCsr;
	uint16 SegCs;
	uint16 SegDs;
	uint16 SegEs;
	uint16 SegFs;
	uint16 SegGs;
	uint16 SegSs;
	uint32 EFlags;
	uint64 Dr0;
	uint64 Dr1;
	uint64 Dr2;
	uint64 Dr3;
	uint64 Dr6;
	uint64 Dr7;
	uint64 Rax;
	uint64 Rcx;
	uint64 Rdx;
	uint64 Rbx;
	uint64 Rsp;
	uint64 Rbp;
	uint64 Rsi;
	uint64 Rdi;
	uint64 R8;
	uint64 R9;
	uint64 R10;
	uint64 R11;
	uint64 R12;
	uint64 R13;
	uint64 R14;
	uint64 R15;
	uint64 Rip;
	byte Pad_godefs_0[512];
	M128a VectorRegister[26];
	uint64 VectorControl;
	uint64 DebugControl;
	uint64 LastBranchToRip;
	uint64 LastBranchFromRip;
	uint64 LastExceptionToRip;
	uint64 LastExceptionFromRip;
};
#pragma pack off
