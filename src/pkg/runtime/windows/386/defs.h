// c:\Users\Hector\Code\go\bin\godefs.exe defs.c

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
	CONTEXT_CONTROL = 0x10001,
	CONTEXT_FULL = 0x10007,
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
	uint32 dwActiveProcessorMask;
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
	uint32 ExceptionInformation[15];
};

typedef struct FloatingSaveArea FloatingSaveArea;
struct FloatingSaveArea {
	uint32 ControlWord;
	uint32 StatusWord;
	uint32 TagWord;
	uint32 ErrorOffset;
	uint32 ErrorSelector;
	uint32 DataOffset;
	uint32 DataSelector;
	uint8 RegisterArea[80];
	uint32 Cr0NpxState;
};

typedef struct Context Context;
struct Context {
	uint32 ContextFlags;
	uint32 Dr0;
	uint32 Dr1;
	uint32 Dr2;
	uint32 Dr3;
	uint32 Dr6;
	uint32 Dr7;
	FloatingSaveArea FloatSave;
	uint32 SegGs;
	uint32 SegFs;
	uint32 SegEs;
	uint32 SegDs;
	uint32 Edi;
	uint32 Esi;
	uint32 Ebx;
	uint32 Edx;
	uint32 Ecx;
	uint32 Eax;
	uint32 Ebp;
	uint32 Eip;
	uint32 SegCs;
	uint32 EFlags;
	uint32 Esp;
	uint32 SegSs;
	uint8 ExtendedRegisters[512];
};
#pragma pack off
