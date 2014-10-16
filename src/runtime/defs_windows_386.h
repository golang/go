// Created by cgo -cdefs - DO NOT EDIT
// cgo -cdefs defs_windows.go


enum {
	PROT_NONE	= 0,
	PROT_READ	= 1,
	PROT_WRITE	= 2,
	PROT_EXEC	= 4,

	MAP_ANON	= 1,
	MAP_PRIVATE	= 2,

	DUPLICATE_SAME_ACCESS	= 0x2,
	THREAD_PRIORITY_HIGHEST	= 0x2,

	SIGINT			= 0x2,
	CTRL_C_EVENT		= 0x0,
	CTRL_BREAK_EVENT	= 0x1,

	CONTEXT_CONTROL	= 0x10001,
	CONTEXT_FULL	= 0x10007,

	EXCEPTION_ACCESS_VIOLATION	= 0xc0000005,
	EXCEPTION_BREAKPOINT		= 0x80000003,
	EXCEPTION_FLT_DENORMAL_OPERAND	= 0xc000008d,
	EXCEPTION_FLT_DIVIDE_BY_ZERO	= 0xc000008e,
	EXCEPTION_FLT_INEXACT_RESULT	= 0xc000008f,
	EXCEPTION_FLT_OVERFLOW		= 0xc0000091,
	EXCEPTION_FLT_UNDERFLOW		= 0xc0000093,
	EXCEPTION_INT_DIVIDE_BY_ZERO	= 0xc0000094,
	EXCEPTION_INT_OVERFLOW		= 0xc0000095,

	INFINITE	= 0xffffffff,
	WAIT_TIMEOUT	= 0x102,

	EXCEPTION_CONTINUE_EXECUTION	= -0x1,
	EXCEPTION_CONTINUE_SEARCH	= 0x0,
};

typedef struct SystemInfo SystemInfo;
typedef struct ExceptionRecord ExceptionRecord;
typedef struct FloatingSaveArea FloatingSaveArea;
typedef struct M128a M128a;
typedef struct Context Context;
typedef struct Overlapped Overlapped;

#pragma pack on

struct SystemInfo {
	byte	anon0[4];
	uint32	dwPageSize;
	byte	*lpMinimumApplicationAddress;
	byte	*lpMaximumApplicationAddress;
	uint32	dwActiveProcessorMask;
	uint32	dwNumberOfProcessors;
	uint32	dwProcessorType;
	uint32	dwAllocationGranularity;
	uint16	wProcessorLevel;
	uint16	wProcessorRevision;
};
struct ExceptionRecord {
	uint32	ExceptionCode;
	uint32	ExceptionFlags;
	ExceptionRecord	*ExceptionRecord;
	byte	*ExceptionAddress;
	uint32	NumberParameters;
	uint32	ExceptionInformation[15];
};
struct FloatingSaveArea {
	uint32	ControlWord;
	uint32	StatusWord;
	uint32	TagWord;
	uint32	ErrorOffset;
	uint32	ErrorSelector;
	uint32	DataOffset;
	uint32	DataSelector;
	uint8	RegisterArea[80];
	uint32	Cr0NpxState;
};
struct Context {
	uint32	ContextFlags;
	uint32	Dr0;
	uint32	Dr1;
	uint32	Dr2;
	uint32	Dr3;
	uint32	Dr6;
	uint32	Dr7;
	FloatingSaveArea	FloatSave;
	uint32	SegGs;
	uint32	SegFs;
	uint32	SegEs;
	uint32	SegDs;
	uint32	Edi;
	uint32	Esi;
	uint32	Ebx;
	uint32	Edx;
	uint32	Ecx;
	uint32	Eax;
	uint32	Ebp;
	uint32	Eip;
	uint32	SegCs;
	uint32	EFlags;
	uint32	Esp;
	uint32	SegSs;
	uint8	ExtendedRegisters[512];
};
struct Overlapped {
	uint32	Internal;
	uint32	InternalHigh;
	byte	anon0[8];
	byte	*hEvent;
};


#pragma pack off
