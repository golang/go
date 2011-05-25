// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

typedef struct {
	uint16 Machine;
	uint16 NumberOfSections;
	uint32 TimeDateStamp;
	uint32 PointerToSymbolTable;
	uint32 NumberOfSymbols;
	uint16 SizeOfOptionalHeader;
	uint16 Characteristics;
} IMAGE_FILE_HEADER;

typedef struct {
	uint32 VirtualAddress;
	uint32 Size;
} IMAGE_DATA_DIRECTORY;

typedef struct {
	uint16 Magic;
	uint8  MajorLinkerVersion;
	uint8  MinorLinkerVersion;
	uint32 SizeOfCode;
	uint32 SizeOfInitializedData;
	uint32 SizeOfUninitializedData;
	uint32 AddressOfEntryPoint;
	uint32 BaseOfCode;
	uint32 BaseOfData;
	uint32 ImageBase;
	uint32 SectionAlignment;
	uint32 FileAlignment;
	uint16 MajorOperatingSystemVersion;
	uint16 MinorOperatingSystemVersion;
	uint16 MajorImageVersion;
	uint16 MinorImageVersion;
	uint16 MajorSubsystemVersion;
	uint16 MinorSubsystemVersion;
	uint32 Win32VersionValue;
	uint32 SizeOfImage;
	uint32 SizeOfHeaders;
	uint32 CheckSum;
	uint16 Subsystem;
	uint16 DllCharacteristics;
	uint32 SizeOfStackReserve;
	uint32 SizeOfStackCommit;
	uint32 SizeOfHeapReserve;
	uint32 SizeOfHeapCommit;
	uint32 LoaderFlags;
	uint32 NumberOfRvaAndSizes;
	IMAGE_DATA_DIRECTORY DataDirectory[16];
} IMAGE_OPTIONAL_HEADER;

typedef struct {
	uint8  Name[8];
	uint32 VirtualSize;
	uint32 VirtualAddress;
	uint32 SizeOfRawData;
	uint32 PointerToRawData;
	uint32 PointerToRelocations;
	uint32 PointerToLineNumbers;
	uint16 NumberOfRelocations;
	uint16 NumberOfLineNumbers;
	uint32 Characteristics;
} IMAGE_SECTION_HEADER;

typedef struct {
	uint32 OriginalFirstThunk;
	uint32 TimeDateStamp;
	uint32 ForwarderChain;
	uint32 Name;
	uint32 FirstThunk;
} IMAGE_IMPORT_DESCRIPTOR;

typedef struct _IMAGE_EXPORT_DIRECTORY {
	uint32 Characteristics;
	uint32 TimeDateStamp;
	uint16 MajorVersion;
	uint16 MinorVersion;
	uint32 Name;
	uint32 Base;
	uint32 NumberOfFunctions;
	uint32 NumberOfNames;
	uint32 AddressOfFunctions;
	uint32 AddressOfNames;
	uint32 AddressOfNameOrdinals;
} IMAGE_EXPORT_DIRECTORY;

#define PEBASE		0x00400000
// SectionAlignment must be greater than or equal to FileAlignment.
// The default is the page size for the architecture.
#define PESECTALIGN	0x1000
// FileAlignment should be a power of 2 between 512 and 64 K, inclusive.
// The default is 512. If the SectionAlignment is less than
// the architecture's page size, then FileAlignment must match SectionAlignment.
#define PEFILEALIGN	(2<<8)
extern	int32	PESECTHEADR;
extern	int32	PEFILEHEADR;

enum {
	IMAGE_FILE_MACHINE_I386 = 0x14c,
	IMAGE_FILE_MACHINE_AMD64 = 0x8664,

	IMAGE_FILE_RELOCS_STRIPPED = 0x0001,
	IMAGE_FILE_EXECUTABLE_IMAGE = 0x0002,
	IMAGE_FILE_LARGE_ADDRESS_AWARE = 0x0020,
	IMAGE_FILE_32BIT_MACHINE = 0x0100,
	IMAGE_FILE_DEBUG_STRIPPED = 0x0200,

	IMAGE_SCN_CNT_CODE = 0x00000020,
	IMAGE_SCN_CNT_INITIALIZED_DATA = 0x00000040,
	IMAGE_SCN_CNT_UNINITIALIZED_DATA = 0x00000080,
	IMAGE_SCN_MEM_EXECUTE = 0x20000000,
	IMAGE_SCN_MEM_READ = 0x40000000,
	IMAGE_SCN_MEM_WRITE = 0x80000000,
	IMAGE_SCN_MEM_DISCARDABLE = 0x2000000,

	IMAGE_DIRECTORY_ENTRY_EXPORT = 0,
	IMAGE_DIRECTORY_ENTRY_IMPORT = 1,
	IMAGE_DIRECTORY_ENTRY_RESOURCE = 2,
	IMAGE_DIRECTORY_ENTRY_EXCEPTION = 3,
	IMAGE_DIRECTORY_ENTRY_SECURITY = 4,
	IMAGE_DIRECTORY_ENTRY_BASERELOC = 5,
	IMAGE_DIRECTORY_ENTRY_DEBUG = 6,
	IMAGE_DIRECTORY_ENTRY_COPYRIGHT = 7,
	IMAGE_DIRECTORY_ENTRY_ARCHITECTURE = 7,
	IMAGE_DIRECTORY_ENTRY_GLOBALPTR = 8,
	IMAGE_DIRECTORY_ENTRY_TLS = 9,
	IMAGE_DIRECTORY_ENTRY_LOAD_CONFIG = 10,
	IMAGE_DIRECTORY_ENTRY_BOUND_IMPORT = 11,
	IMAGE_DIRECTORY_ENTRY_IAT = 12,
	IMAGE_DIRECTORY_ENTRY_DELAY_IMPORT = 13,
	IMAGE_DIRECTORY_ENTRY_COM_DESCRIPTOR = 14,

	IMAGE_SUBSYSTEM_WINDOWS_GUI = 2,
	IMAGE_SUBSYSTEM_WINDOWS_CUI = 3,
};

void peinit(void);
void asmbpe(void);
void dope(void);

IMAGE_SECTION_HEADER* newPEDWARFSection(char *name, vlong size);

// X64
typedef struct {
	uint16 Magic;
	uint8  MajorLinkerVersion;
	uint8  MinorLinkerVersion;
	uint32 SizeOfCode;
	uint32 SizeOfInitializedData;
	uint32 SizeOfUninitializedData;
	uint32 AddressOfEntryPoint;
	uint32 BaseOfCode;
	uint64 ImageBase;
	uint32 SectionAlignment;
	uint32 FileAlignment;
	uint16 MajorOperatingSystemVersion;
	uint16 MinorOperatingSystemVersion;
	uint16 MajorImageVersion;
	uint16 MinorImageVersion;
	uint16 MajorSubsystemVersion;
	uint16 MinorSubsystemVersion;
	uint32 Win32VersionValue;
	uint32 SizeOfImage;
	uint32 SizeOfHeaders;
	uint32 CheckSum;
	uint16 Subsystem;
	uint16 DllCharacteristics;
	uint64 SizeOfStackReserve;
	uint64 SizeOfStackCommit;
	uint64 SizeOfHeapReserve;
	uint64 SizeOfHeapCommit;
	uint32 LoaderFlags;
	uint32 NumberOfRvaAndSizes;
	IMAGE_DATA_DIRECTORY DataDirectory[16];
} PE64_IMAGE_OPTIONAL_HEADER;

void setpersrc(Sym *sym);
