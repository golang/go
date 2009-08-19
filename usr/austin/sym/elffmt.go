// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package sym

import "fmt";

/*
 * ELF64 file format
 */

type elf64Addr uint64
type elf64Off uint64

type elf64Ehdr struct {
//	Ident [elfIdentLen]uint8;	// ELF identification
	Type uint16;			// Object file type
	Machine uint16;			// Machine type
	Version uint32;			// Object file version
	Entry elf64Addr;		// Entry point address
	Phoff elf64Off;			// Program header offset
	Shoff elf64Off;			// Section header offset
	Flags uint32;			// Processor-specific flags
	Ehsize uint16;			// ELF header size
	Phentsize uint16;		// Size of program header entry
	Phnum uint16;			// Number of program header entries
	Shentsize uint16;		// Size of section header entry
	Shnum uint16;			// Number of section header entries
	Shstrndx uint16;		// Section name string table indexes
}

const (
	// Ident indexes
	eiMag0 = 0;			// File identification
	eiMag1 = 1;
	eiMag2 = 2;
	eiMag3 = 3;
	eiClass = 4;			// File class
	eiData = 5;			// Data encoding
	eiVersion = 6;			// File version
	eiOsABI = 7;			// OS/ABI identification
	eiABIVersion = 8;		// ABI version
	eiPad = 9;			// Start of padding bytes
	eiNIdent = 16;			// Size of ident

	// Classes
	elfClass32 = 1;			// 32-bit objects
	elfClass64 = 2;			// 64-bit objects

	// Endians
	elfData2LSB = 1;		// Little-endian
	elfData2MSB = 2;		// Big-endian

	// Types
	etNone = 0;			// No file type
	etRel = 1;			// Relocatable object file
	etExec = 2;			// Executable file
	etDyn = 3;			// Shared object file
	etCore = 4;			// Core file
	etLoOS = 0xFE00;		// Environment-specific use
	etHiOS = 0xFEFF;
	etLoProc = 0xFF00;		// Processor-specific use
	etHiProc = 0xFFFF;

	evCurrent = 1;			// Current version of format
)

type elf64Shdr struct {
	Name uint32;			// Section name
	Type uint32;			// Section type
	Flags uint64;			// Section attributes
	Addr elf64Addr;			// Virtual address in memory
	Off elf64Off;			// Offset in file
	Size uint64;			// Size of section
	Link uint32;			// Link to other section
	Info uint32;			// Miscellaneous information
	Addralign uint64;		// Address alignment boundary
	Entsize uint64;			// Size of entries, if section has table
}

const (
	// Section indices
	shnUndef = 0;			// Used to mark an undefined or meaningless section reference
	shnLoProc = 0xFF00;		// Processor-specific use
	shnHiProc = 0xFF1F;
	shnLoOS = 0xFF20;		// Environment-specific use
	shnHiOS = 0xFF3F;
	shnAbs = 0xFFF1;		// Indicates that the coresponding reference is an absolute value
	shnCommon = 0xFFF2;		// Indicates a symbol that has been declared as a common block

	// Section header types
	shtNull = 0;			// Unused section header
	shtProgBits = 1;		// Information defined by the program
	shtSymTab = 2;			// Linker symbol table
	shtStrTab = 3;			// String table
	shtRela = 4;			// "Rela" type relocation entries
	shtHash = 5;			// Symbol hash table
	shtDynamic = 6;			// Dynamic linking tables
	shtNote = 7;			// Note information
	shtNoBits = 8;			// Uninitialized space; does not occupy any space in the file
	shtRel = 9;			// "Rel" type relocation entries
	shtShlib = 10;			// Reserved
	shtDynSym = 11;			// A dynamic loader symbol table
	shtLoOS = 0x60000000;		// Environment-specific use
	shtHiOS = 0x6FFFFFFF;
	shtLoProc = 0x70000000;		// Processor-specific use
	shtHiProc = 0x7FFFFFFF;

	// Section header flags
	shfWrite = 0x1;			// Writable data
	shfAlloc = 0x2;			// Allocated in memory image of program
	shfExecInstr = 0x4;		// Executable instructions
	shfMaskOS = 0x0F000000;		// Environment-specific use
	shfMaskProc = 0xF0000000;	// Processor-specific use
)

type elf64Phdr struct {
	Type uint32;			// Type of segment
	Flags uint32;			// Segment attributes
	Off elf64Off;			// Offset in file
	Vaddr elf64Addr;		// Virtual address in memory
	Paddr elf64Addr;		// Reserved
	Filesz uint64;			// Size of segment in file
	Memsz uint64;			// Size of segment in memory
	Align uint64;			// Alignment of segment
}

const (
	ptNull = 0;			// Unused entry
	ptLoad = 1;			// Loadable segment
	ptDynamic = 2;			// Dynamic linking tables
	ptInterp = 3;			// Program interpreter path name
	ptNote = 4;			// Note sections
	ptPhdr = 6;			// Program header table

	// Program header flags
	pfX = 0x1;			// Execute permission
	pfW = 0x2;			// Write permission
	pfR = 0x4;			// Read permission
	pfMaskOS = 0x00FF0000;		// Reserved for environment-specific use
	pfMaskProc = 0xFF000000;	// Reserved for processor-specific use
)

/*
 * Exported constants
 */

type ElfType int

const (
	ElfNone ElfType = etNone;
	ElfRel          = etRel;
	ElfExec         = etExec;
	ElfDyn          = etDyn;
	ElfCore         = etCore;
)

type ElfMachine int

const (
	ElfM32	       ElfMachine = 1;
	ElfSPARC       ElfMachine = 2;
	Elf386	       ElfMachine = 3;
	Elf68K	       ElfMachine = 4;
	Elf88K	       ElfMachine = 5;
	Elf860	       ElfMachine = 7;
	ElfMIPS	       ElfMachine = 8;
	ElfS370	       ElfMachine = 9;
	ElfMIPS_RS3_LE ElfMachine = 10;
	ElfPARISC      ElfMachine = 15;
	ElfVPP500      ElfMachine = 17;
	ElfSPARC32PLUS ElfMachine = 18;
	Elf960	       ElfMachine = 19;
	ElfPPC	       ElfMachine = 20;
	ElfPPC64       ElfMachine = 21;
	ElfS390	       ElfMachine = 22;
	ElfV800	       ElfMachine = 36;
	ElfFR20	       ElfMachine = 37;
	ElfRH32	       ElfMachine = 38;
	ElfRCE	       ElfMachine = 39;
	ElfARM	       ElfMachine = 40;
	ElfFAKE_ALPHA  ElfMachine = 41;
	ElfSH	       ElfMachine = 42;
	ElfSPARCV9     ElfMachine = 43;
	ElfTRICORE     ElfMachine = 44;
	ElfARC	       ElfMachine = 45;
	ElfH8_300      ElfMachine = 46;
	ElfH8_300H     ElfMachine = 47;
	ElfH8S	       ElfMachine = 48;
	ElfH8_500      ElfMachine = 49;
	ElfIA_64       ElfMachine = 50;
	ElfMIPS_X      ElfMachine = 51;
	ElfCOLDFIRE    ElfMachine = 52;
	Elf68HC12      ElfMachine = 53;
	ElfMMA	       ElfMachine = 54;
	ElfPCP	       ElfMachine = 55;
	ElfNCPU	       ElfMachine = 56;
	ElfNDR1	       ElfMachine = 57;
	ElfSTARCORE    ElfMachine = 58;
	ElfME16	       ElfMachine = 59;
	ElfST100       ElfMachine = 60;
	ElfTINYJ       ElfMachine = 61;
	ElfX86_64      ElfMachine = 62;
	ElfPDSP	       ElfMachine = 63;
	ElfFX66	       ElfMachine = 66;
	ElfST9PLUS     ElfMachine = 67;
	ElfST7	       ElfMachine = 68;
	Elf68HC16      ElfMachine = 69;
	Elf68HC11      ElfMachine = 70;
	Elf68HC08      ElfMachine = 71;
	Elf68HC05      ElfMachine = 72;
	ElfSVX	       ElfMachine = 73;
	ElfST19	       ElfMachine = 74;
	ElfVAX	       ElfMachine = 75;
	ElfCRIS	       ElfMachine = 76;
	ElfJAVELIN     ElfMachine = 77;
	ElfFIREPATH    ElfMachine = 78;
	ElfZSP	       ElfMachine = 79;
	ElfMMIX	       ElfMachine = 80;
	ElfHUANY       ElfMachine = 81;
	ElfPRISM       ElfMachine = 82;
	ElfAVR	       ElfMachine = 83;
	ElfFR30	       ElfMachine = 84;
	ElfD10V	       ElfMachine = 85;
	ElfD30V	       ElfMachine = 86;
	ElfV850	       ElfMachine = 87;
	ElfM32R	       ElfMachine = 88;
	ElfMN10300     ElfMachine = 89;
	ElfMN10200     ElfMachine = 90;
	ElfPJ	       ElfMachine = 91;
	ElfOPENRISC    ElfMachine = 92;
	ElfARC_A5      ElfMachine = 93;
	ElfXTENSA      ElfMachine = 94;
)

func (m ElfMachine) String() string {
	switch m {
	case ElfMachine(0):
		return "No machine";
	case ElfM32:
		return "AT&T WE 32100";
	case ElfSPARC:
		return "SUN SPARC";
	case Elf386:
		return "Intel 80386";
	case Elf68K:
		return "Motorola m68k family";
	case Elf88K:
		return "Motorola m88k family";
	case Elf860:
		return "Intel 80860";
	case ElfMIPS:
		return "MIPS R3000 big-endian";
	case ElfS370:
		return "IBM System/370";
	case ElfMIPS_RS3_LE:
		return "MIPS R3000 little-endian";
	case ElfPARISC:
		return "HPPA";
	case ElfVPP500:
		return "Fujitsu VPP500";
	case ElfSPARC32PLUS:
		return "Sun's \"v8plus\"";
	case Elf960:
		return "Intel 80960";
	case ElfPPC:
		return "PowerPC";
	case ElfPPC64:
		return "PowerPC 64-bit";
	case ElfS390:
		return "IBM S390";
	case ElfV800:
		return "NEC V800 series";
	case ElfFR20:
		return "Fujitsu FR20";
	case ElfRH32:
		return "TRW RH-32";
	case ElfRCE:
		return "Motorola RCE";
	case ElfARM:
		return "ARM";
	case ElfFAKE_ALPHA:
		return "Digital Alpha";
	case ElfSH:
		return "Hitachi SH";
	case ElfSPARCV9:
		return "SPARC v9 64-bit";
	case ElfTRICORE:
		return "Siemens Tricore";
	case ElfARC:
		return "Argonaut RISC Core";
	case ElfH8_300:
		return "Hitachi H8/300";
	case ElfH8_300H:
		return "Hitachi H8/300H";
	case ElfH8S:
		return "Hitachi H8S";
	case ElfH8_500:
		return "Hitachi H8/500";
	case ElfIA_64:
		return "Intel Merced";
	case ElfMIPS_X:
		return "Stanford MIPS-X";
	case ElfCOLDFIRE:
		return "Motorola Coldfire";
	case Elf68HC12:
		return "Motorola M68HC12";
	case ElfMMA:
		return "Fujitsu MMA Multimedia Accelerato";
	case ElfPCP:
		return "Siemens PCP";
	case ElfNCPU:
		return "Sony nCPU embeeded RISC";
	case ElfNDR1:
		return "Denso NDR1 microprocessor";
	case ElfSTARCORE:
		return "Motorola Start*Core processor";
	case ElfME16:
		return "Toyota ME16 processor";
	case ElfST100:
		return "STMicroelectronic ST100 processor";
	case ElfTINYJ:
		return "Advanced Logic Corp. Tinyj emb.fa";
	case ElfX86_64:
		return "AMD x86-64 architecture";
	case ElfPDSP:
		return "Sony DSP Processor";
	case ElfFX66:
		return "Siemens FX66 microcontroller";
	case ElfST9PLUS:
		return "STMicroelectronics ST9+ 8/16 mc";
	case ElfST7:
		return "STmicroelectronics ST7 8 bit mc";
	case Elf68HC16:
		return "Motorola MC68HC16 microcontroller";
	case Elf68HC11:
		return "Motorola MC68HC11 microcontroller";
	case Elf68HC08:
		return "Motorola MC68HC08 microcontroller";
	case Elf68HC05:
		return "Motorola MC68HC05 microcontroller";
	case ElfSVX:
		return "Silicon Graphics SVx";
	case ElfST19:
		return "STMicroelectronics ST19 8 bit mc";
	case ElfVAX:
		return "Digital VAX";
	case ElfCRIS:
		return "Axis Communications 32-bit embedded processor";
	case ElfJAVELIN:
		return "Infineon Technologies 32-bit embedded processor";
	case ElfFIREPATH:
		return "Element 14 64-bit DSP Processor";
	case ElfZSP:
		return "LSI Logic 16-bit DSP Processor";
	case ElfMMIX:
		return "Donald Knuth's educational 64-bit processor";
	case ElfHUANY:
		return "Harvard University machine-independent object files";
	case ElfPRISM:
		return "SiTera Prism";
	case ElfAVR:
		return "Atmel AVR 8-bit microcontroller";
	case ElfFR30:
		return "Fujitsu FR30";
	case ElfD10V:
		return "Mitsubishi D10V";
	case ElfD30V:
		return "Mitsubishi D30V";
	case ElfV850:
		return "NEC v850";
	case ElfM32R:
		return "Mitsubishi M32R";
	case ElfMN10300:
		return "Matsushita MN10300";
	case ElfMN10200:
		return "Matsushita MN10200";
	case ElfPJ:
		return "picoJava";
	case ElfOPENRISC:
		return "OpenRISC 32-bit embedded processor";
	case ElfARC_A5:
		return "ARC Cores Tangent-A5";
	case ElfXTENSA:
		return "Tensilica Xtensa Architecture";
	}
	return fmt.Sprintf("<unknown %#x>", m);
}
