// g:\opensource\go\bin\godefs.exe -f -m64 defs.c

// MACHINE GENERATED - DO NOT EDIT.

// Constants
enum {
	PROT_NONE = 0,
	PROT_READ = 0x1,
	PROT_WRITE = 0x2,
	PROT_EXEC = 0x4,
	MAP_ANON = 0x1,
	MAP_PRIVATE = 0x2,
	SIGINT = 0x2,
	CTRL_C_EVENT = 0,
	CTRL_BREAK_EVENT = 0x1,
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
#pragma pack off
