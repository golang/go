// godefs -carm-gcc -f -I/usr/local/google/src/linux-2.6.28/arch/arm/include -f -I/usr/local/google/src/linux-2.6.28/include defs_arm.c

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
	int32 tv_sec;
	int32 tv_nsec;
};
#pragma pack off
