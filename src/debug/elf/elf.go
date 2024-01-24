/*
 * ELF constants and data structures
 *
 * Derived from:
 * $FreeBSD: src/sys/sys/elf32.h,v 1.8.14.1 2005/12/30 22:13:58 marcel Exp $
 * $FreeBSD: src/sys/sys/elf64.h,v 1.10.14.1 2005/12/30 22:13:58 marcel Exp $
 * $FreeBSD: src/sys/sys/elf_common.h,v 1.15.8.1 2005/12/30 22:13:58 marcel Exp $
 * $FreeBSD: src/sys/alpha/include/elf.h,v 1.14 2003/09/25 01:10:22 peter Exp $
 * $FreeBSD: src/sys/amd64/include/elf.h,v 1.18 2004/08/03 08:21:48 dfr Exp $
 * $FreeBSD: src/sys/arm/include/elf.h,v 1.5.2.1 2006/06/30 21:42:52 cognet Exp $
 * $FreeBSD: src/sys/i386/include/elf.h,v 1.16 2004/08/02 19:12:17 dfr Exp $
 * $FreeBSD: src/sys/powerpc/include/elf.h,v 1.7 2004/11/02 09:47:01 ssouhlal Exp $
 * $FreeBSD: src/sys/sparc64/include/elf.h,v 1.12 2003/09/25 01:10:26 peter Exp $
 * "System V ABI" (http://www.sco.com/developers/gabi/latest/ch4.eheader.html)
 * "ELF for the ARM® 64-bit Architecture (AArch64)" (ARM IHI 0056B)
 * "RISC-V ELF psABI specification" (https://github.com/riscv-non-isa/riscv-elf-psabi-doc/blob/master/riscv-elf.adoc)
 * llvm/BinaryFormat/ELF.h - ELF constants and structures
 *
 * Copyright (c) 1996-1998 John D. Polstra.  All rights reserved.
 * Copyright (c) 2001 David E. O'Brien
 * Portions Copyright 2009 The Go Authors. All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions
 * are met:
 * 1. Redistributions of source code must retain the above copyright
 *    notice, this list of conditions and the following disclaimer.
 * 2. Redistributions in binary form must reproduce the above copyright
 *    notice, this list of conditions and the following disclaimer in the
 *    documentation and/or other materials provided with the distribution.
 *
 * THIS SOFTWARE IS PROVIDED BY THE AUTHOR AND CONTRIBUTORS ``AS IS'' AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED.  IN NO EVENT SHALL THE AUTHOR OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS
 * OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION)
 * HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
 * LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY
 * OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF
 * SUCH DAMAGE.
 */

package elf

import "strconv"

/*
 * Constants
 */

// Indexes into the Header.Ident array.
const (
	EI_CLASS      = 4  /* Class of machine. */
	EI_DATA       = 5  /* Data format. */
	EI_VERSION    = 6  /* ELF format version. */
	EI_OSABI      = 7  /* Operating system / ABI identification */
	EI_ABIVERSION = 8  /* ABI version */
	EI_PAD        = 9  /* Start of padding (per SVR4 ABI). */
	EI_NIDENT     = 16 /* Size of e_ident array. */
)

// Initial magic number for ELF files.
const ELFMAG = "\177ELF"

// Version is found in Header.Ident[EI_VERSION] and Header.Version.
type Version byte

const (
	EV_NONE    Version = 0
	EV_CURRENT Version = 1
)

var versionStrings = []intName{
	{0, "EV_NONE"},
	{1, "EV_CURRENT"},
}

func (i Version) String() string   { return stringName(uint32(i), versionStrings, false) }
func (i Version) GoString() string { return stringName(uint32(i), versionStrings, true) }

// Class is found in Header.Ident[EI_CLASS] and Header.Class.
type Class byte

const (
	ELFCLASSNONE Class = 0 /* Unknown class. */
	ELFCLASS32   Class = 1 /* 32-bit architecture. */
	ELFCLASS64   Class = 2 /* 64-bit architecture. */
)

var classStrings = []intName{
	{0, "ELFCLASSNONE"},
	{1, "ELFCLASS32"},
	{2, "ELFCLASS64"},
}

func (i Class) String() string   { return stringName(uint32(i), classStrings, false) }
func (i Class) GoString() string { return stringName(uint32(i), classStrings, true) }

// Data is found in Header.Ident[EI_DATA] and Header.Data.
type Data byte

const (
	ELFDATANONE Data = 0 /* Unknown data format. */
	ELFDATA2LSB Data = 1 /* 2's complement little-endian. */
	ELFDATA2MSB Data = 2 /* 2's complement big-endian. */
)

var dataStrings = []intName{
	{0, "ELFDATANONE"},
	{1, "ELFDATA2LSB"},
	{2, "ELFDATA2MSB"},
}

func (i Data) String() string   { return stringName(uint32(i), dataStrings, false) }
func (i Data) GoString() string { return stringName(uint32(i), dataStrings, true) }

// OSABI is found in Header.Ident[EI_OSABI] and Header.OSABI.
type OSABI byte

const (
	ELFOSABI_NONE       OSABI = 0   /* UNIX System V ABI */
	ELFOSABI_HPUX       OSABI = 1   /* HP-UX operating system */
	ELFOSABI_NETBSD     OSABI = 2   /* NetBSD */
	ELFOSABI_LINUX      OSABI = 3   /* Linux */
	ELFOSABI_HURD       OSABI = 4   /* Hurd */
	ELFOSABI_86OPEN     OSABI = 5   /* 86Open common IA32 ABI */
	ELFOSABI_SOLARIS    OSABI = 6   /* Solaris */
	ELFOSABI_AIX        OSABI = 7   /* AIX */
	ELFOSABI_IRIX       OSABI = 8   /* IRIX */
	ELFOSABI_FREEBSD    OSABI = 9   /* FreeBSD */
	ELFOSABI_TRU64      OSABI = 10  /* TRU64 UNIX */
	ELFOSABI_MODESTO    OSABI = 11  /* Novell Modesto */
	ELFOSABI_OPENBSD    OSABI = 12  /* OpenBSD */
	ELFOSABI_OPENVMS    OSABI = 13  /* Open VMS */
	ELFOSABI_NSK        OSABI = 14  /* HP Non-Stop Kernel */
	ELFOSABI_AROS       OSABI = 15  /* Amiga Research OS */
	ELFOSABI_FENIXOS    OSABI = 16  /* The FenixOS highly scalable multi-core OS */
	ELFOSABI_CLOUDABI   OSABI = 17  /* Nuxi CloudABI */
	ELFOSABI_ARM        OSABI = 97  /* ARM */
	ELFOSABI_STANDALONE OSABI = 255 /* Standalone (embedded) application */
)

var osabiStrings = []intName{
	{0, "ELFOSABI_NONE"},
	{1, "ELFOSABI_HPUX"},
	{2, "ELFOSABI_NETBSD"},
	{3, "ELFOSABI_LINUX"},
	{4, "ELFOSABI_HURD"},
	{5, "ELFOSABI_86OPEN"},
	{6, "ELFOSABI_SOLARIS"},
	{7, "ELFOSABI_AIX"},
	{8, "ELFOSABI_IRIX"},
	{9, "ELFOSABI_FREEBSD"},
	{10, "ELFOSABI_TRU64"},
	{11, "ELFOSABI_MODESTO"},
	{12, "ELFOSABI_OPENBSD"},
	{13, "ELFOSABI_OPENVMS"},
	{14, "ELFOSABI_NSK"},
	{15, "ELFOSABI_AROS"},
	{16, "ELFOSABI_FENIXOS"},
	{17, "ELFOSABI_CLOUDABI"},
	{97, "ELFOSABI_ARM"},
	{255, "ELFOSABI_STANDALONE"},
}

func (i OSABI) String() string   { return stringName(uint32(i), osabiStrings, false) }
func (i OSABI) GoString() string { return stringName(uint32(i), osabiStrings, true) }

// Type is found in Header.Type.
type Type uint16

const (
	ET_NONE   Type = 0      /* Unknown type. */
	ET_REL    Type = 1      /* Relocatable. */
	ET_EXEC   Type = 2      /* Executable. */
	ET_DYN    Type = 3      /* Shared object. */
	ET_CORE   Type = 4      /* Core file. */
	ET_LOOS   Type = 0xfe00 /* First operating system specific. */
	ET_HIOS   Type = 0xfeff /* Last operating system-specific. */
	ET_LOPROC Type = 0xff00 /* First processor-specific. */
	ET_HIPROC Type = 0xffff /* Last processor-specific. */
)

var typeStrings = []intName{
	{0, "ET_NONE"},
	{1, "ET_REL"},
	{2, "ET_EXEC"},
	{3, "ET_DYN"},
	{4, "ET_CORE"},
	{0xfe00, "ET_LOOS"},
	{0xfeff, "ET_HIOS"},
	{0xff00, "ET_LOPROC"},
	{0xffff, "ET_HIPROC"},
}

func (i Type) String() string   { return stringName(uint32(i), typeStrings, false) }
func (i Type) GoString() string { return stringName(uint32(i), typeStrings, true) }

// Machine is found in Header.Machine.
type Machine uint16

const (
	EM_NONE          Machine = 0   /* Unknown machine. */
	EM_M32           Machine = 1   /* AT&T WE32100. */
	EM_SPARC         Machine = 2   /* Sun SPARC. */
	EM_386           Machine = 3   /* Intel i386. */
	EM_68K           Machine = 4   /* Motorola 68000. */
	EM_88K           Machine = 5   /* Motorola 88000. */
	EM_860           Machine = 7   /* Intel i860. */
	EM_MIPS          Machine = 8   /* MIPS R3000 Big-Endian only. */
	EM_S370          Machine = 9   /* IBM System/370. */
	EM_MIPS_RS3_LE   Machine = 10  /* MIPS R3000 Little-Endian. */
	EM_PARISC        Machine = 15  /* HP PA-RISC. */
	EM_VPP500        Machine = 17  /* Fujitsu VPP500. */
	EM_SPARC32PLUS   Machine = 18  /* SPARC v8plus. */
	EM_960           Machine = 19  /* Intel 80960. */
	EM_PPC           Machine = 20  /* PowerPC 32-bit. */
	EM_PPC64         Machine = 21  /* PowerPC 64-bit. */
	EM_S390          Machine = 22  /* IBM System/390. */
	EM_V800          Machine = 36  /* NEC V800. */
	EM_FR20          Machine = 37  /* Fujitsu FR20. */
	EM_RH32          Machine = 38  /* TRW RH-32. */
	EM_RCE           Machine = 39  /* Motorola RCE. */
	EM_ARM           Machine = 40  /* ARM. */
	EM_SH            Machine = 42  /* Hitachi SH. */
	EM_SPARCV9       Machine = 43  /* SPARC v9 64-bit. */
	EM_TRICORE       Machine = 44  /* Siemens TriCore embedded processor. */
	EM_ARC           Machine = 45  /* Argonaut RISC Core. */
	EM_H8_300        Machine = 46  /* Hitachi H8/300. */
	EM_H8_300H       Machine = 47  /* Hitachi H8/300H. */
	EM_H8S           Machine = 48  /* Hitachi H8S. */
	EM_H8_500        Machine = 49  /* Hitachi H8/500. */
	EM_IA_64         Machine = 50  /* Intel IA-64 Processor. */
	EM_MIPS_X        Machine = 51  /* Stanford MIPS-X. */
	EM_COLDFIRE      Machine = 52  /* Motorola ColdFire. */
	EM_68HC12        Machine = 53  /* Motorola M68HC12. */
	EM_MMA           Machine = 54  /* Fujitsu MMA. */
	EM_PCP           Machine = 55  /* Siemens PCP. */
	EM_NCPU          Machine = 56  /* Sony nCPU. */
	EM_NDR1          Machine = 57  /* Denso NDR1 microprocessor. */
	EM_STARCORE      Machine = 58  /* Motorola Star*Core processor. */
	EM_ME16          Machine = 59  /* Toyota ME16 processor. */
	EM_ST100         Machine = 60  /* STMicroelectronics ST100 processor. */
	EM_TINYJ         Machine = 61  /* Advanced Logic Corp. TinyJ processor. */
	EM_X86_64        Machine = 62  /* Advanced Micro Devices x86-64 */
	EM_PDSP          Machine = 63  /* Sony DSP Processor */
	EM_PDP10         Machine = 64  /* Digital Equipment Corp. PDP-10 */
	EM_PDP11         Machine = 65  /* Digital Equipment Corp. PDP-11 */
	EM_FX66          Machine = 66  /* Siemens FX66 microcontroller */
	EM_ST9PLUS       Machine = 67  /* STMicroelectronics ST9+ 8/16 bit microcontroller */
	EM_ST7           Machine = 68  /* STMicroelectronics ST7 8-bit microcontroller */
	EM_68HC16        Machine = 69  /* Motorola MC68HC16 Microcontroller */
	EM_68HC11        Machine = 70  /* Motorola MC68HC11 Microcontroller */
	EM_68HC08        Machine = 71  /* Motorola MC68HC08 Microcontroller */
	EM_68HC05        Machine = 72  /* Motorola MC68HC05 Microcontroller */
	EM_SVX           Machine = 73  /* Silicon Graphics SVx */
	EM_ST19          Machine = 74  /* STMicroelectronics ST19 8-bit microcontroller */
	EM_VAX           Machine = 75  /* Digital VAX */
	EM_CRIS          Machine = 76  /* Axis Communications 32-bit embedded processor */
	EM_JAVELIN       Machine = 77  /* Infineon Technologies 32-bit embedded processor */
	EM_FIREPATH      Machine = 78  /* Element 14 64-bit DSP Processor */
	EM_ZSP           Machine = 79  /* LSI Logic 16-bit DSP Processor */
	EM_MMIX          Machine = 80  /* Donald Knuth's educational 64-bit processor */
	EM_HUANY         Machine = 81  /* Harvard University machine-independent object files */
	EM_PRISM         Machine = 82  /* SiTera Prism */
	EM_AVR           Machine = 83  /* Atmel AVR 8-bit microcontroller */
	EM_FR30          Machine = 84  /* Fujitsu FR30 */
	EM_D10V          Machine = 85  /* Mitsubishi D10V */
	EM_D30V          Machine = 86  /* Mitsubishi D30V */
	EM_V850          Machine = 87  /* NEC v850 */
	EM_M32R          Machine = 88  /* Mitsubishi M32R */
	EM_MN10300       Machine = 89  /* Matsushita MN10300 */
	EM_MN10200       Machine = 90  /* Matsushita MN10200 */
	EM_PJ            Machine = 91  /* picoJava */
	EM_OPENRISC      Machine = 92  /* OpenRISC 32-bit embedded processor */
	EM_ARC_COMPACT   Machine = 93  /* ARC International ARCompact processor (old spelling/synonym: EM_ARC_A5) */
	EM_XTENSA        Machine = 94  /* Tensilica Xtensa Architecture */
	EM_VIDEOCORE     Machine = 95  /* Alphamosaic VideoCore processor */
	EM_TMM_GPP       Machine = 96  /* Thompson Multimedia General Purpose Processor */
	EM_NS32K         Machine = 97  /* National Semiconductor 32000 series */
	EM_TPC           Machine = 98  /* Tenor Network TPC processor */
	EM_SNP1K         Machine = 99  /* Trebia SNP 1000 processor */
	EM_ST200         Machine = 100 /* STMicroelectronics (www.st.com) ST200 microcontroller */
	EM_IP2K          Machine = 101 /* Ubicom IP2xxx microcontroller family */
	EM_MAX           Machine = 102 /* MAX Processor */
	EM_CR            Machine = 103 /* National Semiconductor CompactRISC microprocessor */
	EM_F2MC16        Machine = 104 /* Fujitsu F2MC16 */
	EM_MSP430        Machine = 105 /* Texas Instruments embedded microcontroller msp430 */
	EM_BLACKFIN      Machine = 106 /* Analog Devices Blackfin (DSP) processor */
	EM_SE_C33        Machine = 107 /* S1C33 Family of Seiko Epson processors */
	EM_SEP           Machine = 108 /* Sharp embedded microprocessor */
	EM_ARCA          Machine = 109 /* Arca RISC Microprocessor */
	EM_UNICORE       Machine = 110 /* Microprocessor series from PKU-Unity Ltd. and MPRC of Peking University */
	EM_EXCESS        Machine = 111 /* eXcess: 16/32/64-bit configurable embedded CPU */
	EM_DXP           Machine = 112 /* Icera Semiconductor Inc. Deep Execution Processor */
	EM_ALTERA_NIOS2  Machine = 113 /* Altera Nios II soft-core processor */
	EM_CRX           Machine = 114 /* National Semiconductor CompactRISC CRX microprocessor */
	EM_XGATE         Machine = 115 /* Motorola XGATE embedded processor */
	EM_C166          Machine = 116 /* Infineon C16x/XC16x processor */
	EM_M16C          Machine = 117 /* Renesas M16C series microprocessors */
	EM_DSPIC30F      Machine = 118 /* Microchip Technology dsPIC30F Digital Signal Controller */
	EM_CE            Machine = 119 /* Freescale Communication Engine RISC core */
	EM_M32C          Machine = 120 /* Renesas M32C series microprocessors */
	EM_TSK3000       Machine = 131 /* Altium TSK3000 core */
	EM_RS08          Machine = 132 /* Freescale RS08 embedded processor */
	EM_SHARC         Machine = 133 /* Analog Devices SHARC family of 32-bit DSP processors */
	EM_ECOG2         Machine = 134 /* Cyan Technology eCOG2 microprocessor */
	EM_SCORE7        Machine = 135 /* Sunplus S+core7 RISC processor */
	EM_DSP24         Machine = 136 /* New Japan Radio (NJR) 24-bit DSP Processor */
	EM_VIDEOCORE3    Machine = 137 /* Broadcom VideoCore III processor */
	EM_LATTICEMICO32 Machine = 138 /* RISC processor for Lattice FPGA architecture */
	EM_SE_C17        Machine = 139 /* Seiko Epson C17 family */
	EM_TI_C6000      Machine = 140 /* The Texas Instruments TMS320C6000 DSP family */
	EM_TI_C2000      Machine = 141 /* The Texas Instruments TMS320C2000 DSP family */
	EM_TI_C5500      Machine = 142 /* The Texas Instruments TMS320C55x DSP family */
	EM_TI_ARP32      Machine = 143 /* Texas Instruments Application Specific RISC Processor, 32bit fetch */
	EM_TI_PRU        Machine = 144 /* Texas Instruments Programmable Realtime Unit */
	EM_MMDSP_PLUS    Machine = 160 /* STMicroelectronics 64bit VLIW Data Signal Processor */
	EM_CYPRESS_M8C   Machine = 161 /* Cypress M8C microprocessor */
	EM_R32C          Machine = 162 /* Renesas R32C series microprocessors */
	EM_TRIMEDIA      Machine = 163 /* NXP Semiconductors TriMedia architecture family */
	EM_QDSP6         Machine = 164 /* QUALCOMM DSP6 Processor */
	EM_8051          Machine = 165 /* Intel 8051 and variants */
	EM_STXP7X        Machine = 166 /* STMicroelectronics STxP7x family of configurable and extensible RISC processors */
	EM_NDS32         Machine = 167 /* Andes Technology compact code size embedded RISC processor family */
	EM_ECOG1         Machine = 168 /* Cyan Technology eCOG1X family */
	EM_ECOG1X        Machine = 168 /* Cyan Technology eCOG1X family */
	EM_MAXQ30        Machine = 169 /* Dallas Semiconductor MAXQ30 Core Micro-controllers */
	EM_XIMO16        Machine = 170 /* New Japan Radio (NJR) 16-bit DSP Processor */
	EM_MANIK         Machine = 171 /* M2000 Reconfigurable RISC Microprocessor */
	EM_CRAYNV2       Machine = 172 /* Cray Inc. NV2 vector architecture */
	EM_RX            Machine = 173 /* Renesas RX family */
	EM_METAG         Machine = 174 /* Imagination Technologies META processor architecture */
	EM_MCST_ELBRUS   Machine = 175 /* MCST Elbrus general purpose hardware architecture */
	EM_ECOG16        Machine = 176 /* Cyan Technology eCOG16 family */
	EM_CR16          Machine = 177 /* National Semiconductor CompactRISC CR16 16-bit microprocessor */
	EM_ETPU          Machine = 178 /* Freescale Extended Time Processing Unit */
	EM_SLE9X         Machine = 179 /* Infineon Technologies SLE9X core */
	EM_L10M          Machine = 180 /* Intel L10M */
	EM_K10M          Machine = 181 /* Intel K10M */
	EM_AARCH64       Machine = 183 /* ARM 64-bit Architecture (AArch64) */
	EM_AVR32         Machine = 185 /* Atmel Corporation 32-bit microprocessor family */
	EM_STM8          Machine = 186 /* STMicroeletronics STM8 8-bit microcontroller */
	EM_TILE64        Machine = 187 /* Tilera TILE64 multicore architecture family */
	EM_TILEPRO       Machine = 188 /* Tilera TILEPro multicore architecture family */
	EM_MICROBLAZE    Machine = 189 /* Xilinx MicroBlaze 32-bit RISC soft processor core */
	EM_CUDA          Machine = 190 /* NVIDIA CUDA architecture */
	EM_TILEGX        Machine = 191 /* Tilera TILE-Gx multicore architecture family */
	EM_CLOUDSHIELD   Machine = 192 /* CloudShield architecture family */
	EM_COREA_1ST     Machine = 193 /* KIPO-KAIST Core-A 1st generation processor family */
	EM_COREA_2ND     Machine = 194 /* KIPO-KAIST Core-A 2nd generation processor family */
	EM_ARC_COMPACT2  Machine = 195 /* Synopsys ARCompact V2 */
	EM_OPEN8         Machine = 196 /* Open8 8-bit RISC soft processor core */
	EM_RL78          Machine = 197 /* Renesas RL78 family */
	EM_VIDEOCORE5    Machine = 198 /* Broadcom VideoCore V processor */
	EM_78KOR         Machine = 199 /* Renesas 78KOR family */
	EM_56800EX       Machine = 200 /* Freescale 56800EX Digital Signal Controller (DSC) */
	EM_BA1           Machine = 201 /* Beyond BA1 CPU architecture */
	EM_BA2           Machine = 202 /* Beyond BA2 CPU architecture */
	EM_XCORE         Machine = 203 /* XMOS xCORE processor family */
	EM_MCHP_PIC      Machine = 204 /* Microchip 8-bit PIC(r) family */
	EM_INTEL205      Machine = 205 /* Reserved by Intel */
	EM_INTEL206      Machine = 206 /* Reserved by Intel */
	EM_INTEL207      Machine = 207 /* Reserved by Intel */
	EM_INTEL208      Machine = 208 /* Reserved by Intel */
	EM_INTEL209      Machine = 209 /* Reserved by Intel */
	EM_KM32          Machine = 210 /* KM211 KM32 32-bit processor */
	EM_KMX32         Machine = 211 /* KM211 KMX32 32-bit processor */
	EM_KMX16         Machine = 212 /* KM211 KMX16 16-bit processor */
	EM_KMX8          Machine = 213 /* KM211 KMX8 8-bit processor */
	EM_KVARC         Machine = 214 /* KM211 KVARC processor */
	EM_CDP           Machine = 215 /* Paneve CDP architecture family */
	EM_COGE          Machine = 216 /* Cognitive Smart Memory Processor */
	EM_COOL          Machine = 217 /* Bluechip Systems CoolEngine */
	EM_NORC          Machine = 218 /* Nanoradio Optimized RISC */
	EM_CSR_KALIMBA   Machine = 219 /* CSR Kalimba architecture family */
	EM_Z80           Machine = 220 /* Zilog Z80 */
	EM_VISIUM        Machine = 221 /* Controls and Data Services VISIUMcore processor */
	EM_FT32          Machine = 222 /* FTDI Chip FT32 high performance 32-bit RISC architecture */
	EM_MOXIE         Machine = 223 /* Moxie processor family */
	EM_AMDGPU        Machine = 224 /* AMD GPU architecture */
	EM_RISCV         Machine = 243 /* RISC-V */
	EM_LANAI         Machine = 244 /* Lanai 32-bit processor */
	EM_BPF           Machine = 247 /* Linux BPF – in-kernel virtual machine */
	EM_LOONGARCH     Machine = 258 /* LoongArch */

	/* Non-standard or deprecated. */
	EM_486         Machine = 6      /* Intel i486. */
	EM_MIPS_RS4_BE Machine = 10     /* MIPS R4000 Big-Endian */
	EM_ALPHA_STD   Machine = 41     /* Digital Alpha (standard value). */
	EM_ALPHA       Machine = 0x9026 /* Alpha (written in the absence of an ABI) */
)

var machineStrings = []intName{
	{0, "EM_NONE"},
	{1, "EM_M32"},
	{2, "EM_SPARC"},
	{3, "EM_386"},
	{4, "EM_68K"},
	{5, "EM_88K"},
	{7, "EM_860"},
	{8, "EM_MIPS"},
	{9, "EM_S370"},
	{10, "EM_MIPS_RS3_LE"},
	{15, "EM_PARISC"},
	{17, "EM_VPP500"},
	{18, "EM_SPARC32PLUS"},
	{19, "EM_960"},
	{20, "EM_PPC"},
	{21, "EM_PPC64"},
	{22, "EM_S390"},
	{36, "EM_V800"},
	{37, "EM_FR20"},
	{38, "EM_RH32"},
	{39, "EM_RCE"},
	{40, "EM_ARM"},
	{42, "EM_SH"},
	{43, "EM_SPARCV9"},
	{44, "EM_TRICORE"},
	{45, "EM_ARC"},
	{46, "EM_H8_300"},
	{47, "EM_H8_300H"},
	{48, "EM_H8S"},
	{49, "EM_H8_500"},
	{50, "EM_IA_64"},
	{51, "EM_MIPS_X"},
	{52, "EM_COLDFIRE"},
	{53, "EM_68HC12"},
	{54, "EM_MMA"},
	{55, "EM_PCP"},
	{56, "EM_NCPU"},
	{57, "EM_NDR1"},
	{58, "EM_STARCORE"},
	{59, "EM_ME16"},
	{60, "EM_ST100"},
	{61, "EM_TINYJ"},
	{62, "EM_X86_64"},
	{63, "EM_PDSP"},
	{64, "EM_PDP10"},
	{65, "EM_PDP11"},
	{66, "EM_FX66"},
	{67, "EM_ST9PLUS"},
	{68, "EM_ST7"},
	{69, "EM_68HC16"},
	{70, "EM_68HC11"},
	{71, "EM_68HC08"},
	{72, "EM_68HC05"},
	{73, "EM_SVX"},
	{74, "EM_ST19"},
	{75, "EM_VAX"},
	{76, "EM_CRIS"},
	{77, "EM_JAVELIN"},
	{78, "EM_FIREPATH"},
	{79, "EM_ZSP"},
	{80, "EM_MMIX"},
	{81, "EM_HUANY"},
	{82, "EM_PRISM"},
	{83, "EM_AVR"},
	{84, "EM_FR30"},
	{85, "EM_D10V"},
	{86, "EM_D30V"},
	{87, "EM_V850"},
	{88, "EM_M32R"},
	{89, "EM_MN10300"},
	{90, "EM_MN10200"},
	{91, "EM_PJ"},
	{92, "EM_OPENRISC"},
	{93, "EM_ARC_COMPACT"},
	{94, "EM_XTENSA"},
	{95, "EM_VIDEOCORE"},
	{96, "EM_TMM_GPP"},
	{97, "EM_NS32K"},
	{98, "EM_TPC"},
	{99, "EM_SNP1K"},
	{100, "EM_ST200"},
	{101, "EM_IP2K"},
	{102, "EM_MAX"},
	{103, "EM_CR"},
	{104, "EM_F2MC16"},
	{105, "EM_MSP430"},
	{106, "EM_BLACKFIN"},
	{107, "EM_SE_C33"},
	{108, "EM_SEP"},
	{109, "EM_ARCA"},
	{110, "EM_UNICORE"},
	{111, "EM_EXCESS"},
	{112, "EM_DXP"},
	{113, "EM_ALTERA_NIOS2"},
	{114, "EM_CRX"},
	{115, "EM_XGATE"},
	{116, "EM_C166"},
	{117, "EM_M16C"},
	{118, "EM_DSPIC30F"},
	{119, "EM_CE"},
	{120, "EM_M32C"},
	{131, "EM_TSK3000"},
	{132, "EM_RS08"},
	{133, "EM_SHARC"},
	{134, "EM_ECOG2"},
	{135, "EM_SCORE7"},
	{136, "EM_DSP24"},
	{137, "EM_VIDEOCORE3"},
	{138, "EM_LATTICEMICO32"},
	{139, "EM_SE_C17"},
	{140, "EM_TI_C6000"},
	{141, "EM_TI_C2000"},
	{142, "EM_TI_C5500"},
	{143, "EM_TI_ARP32"},
	{144, "EM_TI_PRU"},
	{160, "EM_MMDSP_PLUS"},
	{161, "EM_CYPRESS_M8C"},
	{162, "EM_R32C"},
	{163, "EM_TRIMEDIA"},
	{164, "EM_QDSP6"},
	{165, "EM_8051"},
	{166, "EM_STXP7X"},
	{167, "EM_NDS32"},
	{168, "EM_ECOG1"},
	{168, "EM_ECOG1X"},
	{169, "EM_MAXQ30"},
	{170, "EM_XIMO16"},
	{171, "EM_MANIK"},
	{172, "EM_CRAYNV2"},
	{173, "EM_RX"},
	{174, "EM_METAG"},
	{175, "EM_MCST_ELBRUS"},
	{176, "EM_ECOG16"},
	{177, "EM_CR16"},
	{178, "EM_ETPU"},
	{179, "EM_SLE9X"},
	{180, "EM_L10M"},
	{181, "EM_K10M"},
	{183, "EM_AARCH64"},
	{185, "EM_AVR32"},
	{186, "EM_STM8"},
	{187, "EM_TILE64"},
	{188, "EM_TILEPRO"},
	{189, "EM_MICROBLAZE"},
	{190, "EM_CUDA"},
	{191, "EM_TILEGX"},
	{192, "EM_CLOUDSHIELD"},
	{193, "EM_COREA_1ST"},
	{194, "EM_COREA_2ND"},
	{195, "EM_ARC_COMPACT2"},
	{196, "EM_OPEN8"},
	{197, "EM_RL78"},
	{198, "EM_VIDEOCORE5"},
	{199, "EM_78KOR"},
	{200, "EM_56800EX"},
	{201, "EM_BA1"},
	{202, "EM_BA2"},
	{203, "EM_XCORE"},
	{204, "EM_MCHP_PIC"},
	{205, "EM_INTEL205"},
	{206, "EM_INTEL206"},
	{207, "EM_INTEL207"},
	{208, "EM_INTEL208"},
	{209, "EM_INTEL209"},
	{210, "EM_KM32"},
	{211, "EM_KMX32"},
	{212, "EM_KMX16"},
	{213, "EM_KMX8"},
	{214, "EM_KVARC"},
	{215, "EM_CDP"},
	{216, "EM_COGE"},
	{217, "EM_COOL"},
	{218, "EM_NORC"},
	{219, "EM_CSR_KALIMBA "},
	{220, "EM_Z80 "},
	{221, "EM_VISIUM "},
	{222, "EM_FT32 "},
	{223, "EM_MOXIE"},
	{224, "EM_AMDGPU"},
	{243, "EM_RISCV"},
	{244, "EM_LANAI"},
	{247, "EM_BPF"},
	{258, "EM_LOONGARCH"},

	/* Non-standard or deprecated. */
	{6, "EM_486"},
	{10, "EM_MIPS_RS4_BE"},
	{41, "EM_ALPHA_STD"},
	{0x9026, "EM_ALPHA"},
}

func (i Machine) String() string   { return stringName(uint32(i), machineStrings, false) }
func (i Machine) GoString() string { return stringName(uint32(i), machineStrings, true) }

// Special section indices.
type SectionIndex int

const (
	SHN_UNDEF     SectionIndex = 0      /* Undefined, missing, irrelevant. */
	SHN_LORESERVE SectionIndex = 0xff00 /* First of reserved range. */
	SHN_LOPROC    SectionIndex = 0xff00 /* First processor-specific. */
	SHN_HIPROC    SectionIndex = 0xff1f /* Last processor-specific. */
	SHN_LOOS      SectionIndex = 0xff20 /* First operating system-specific. */
	SHN_HIOS      SectionIndex = 0xff3f /* Last operating system-specific. */
	SHN_ABS       SectionIndex = 0xfff1 /* Absolute values. */
	SHN_COMMON    SectionIndex = 0xfff2 /* Common data. */
	SHN_XINDEX    SectionIndex = 0xffff /* Escape; index stored elsewhere. */
	SHN_HIRESERVE SectionIndex = 0xffff /* Last of reserved range. */
)

var shnStrings = []intName{
	{0, "SHN_UNDEF"},
	{0xff00, "SHN_LOPROC"},
	{0xff20, "SHN_LOOS"},
	{0xfff1, "SHN_ABS"},
	{0xfff2, "SHN_COMMON"},
	{0xffff, "SHN_XINDEX"},
}

func (i SectionIndex) String() string   { return stringName(uint32(i), shnStrings, false) }
func (i SectionIndex) GoString() string { return stringName(uint32(i), shnStrings, true) }

// Section type.
type SectionType uint32

const (
	SHT_NULL           SectionType = 0          /* inactive */
	SHT_PROGBITS       SectionType = 1          /* program defined information */
	SHT_SYMTAB         SectionType = 2          /* symbol table section */
	SHT_STRTAB         SectionType = 3          /* string table section */
	SHT_RELA           SectionType = 4          /* relocation section with addends */
	SHT_HASH           SectionType = 5          /* symbol hash table section */
	SHT_DYNAMIC        SectionType = 6          /* dynamic section */
	SHT_NOTE           SectionType = 7          /* note section */
	SHT_NOBITS         SectionType = 8          /* no space section */
	SHT_REL            SectionType = 9          /* relocation section - no addends */
	SHT_SHLIB          SectionType = 10         /* reserved - purpose unknown */
	SHT_DYNSYM         SectionType = 11         /* dynamic symbol table section */
	SHT_INIT_ARRAY     SectionType = 14         /* Initialization function pointers. */
	SHT_FINI_ARRAY     SectionType = 15         /* Termination function pointers. */
	SHT_PREINIT_ARRAY  SectionType = 16         /* Pre-initialization function ptrs. */
	SHT_GROUP          SectionType = 17         /* Section group. */
	SHT_SYMTAB_SHNDX   SectionType = 18         /* Section indexes (see SHN_XINDEX). */
	SHT_LOOS           SectionType = 0x60000000 /* First of OS specific semantics */
	SHT_GNU_ATTRIBUTES SectionType = 0x6ffffff5 /* GNU object attributes */
	SHT_GNU_HASH       SectionType = 0x6ffffff6 /* GNU hash table */
	SHT_GNU_LIBLIST    SectionType = 0x6ffffff7 /* GNU prelink library list */
	SHT_GNU_VERDEF     SectionType = 0x6ffffffd /* GNU version definition section */
	SHT_GNU_VERNEED    SectionType = 0x6ffffffe /* GNU version needs section */
	SHT_GNU_VERSYM     SectionType = 0x6fffffff /* GNU version symbol table */
	SHT_HIOS           SectionType = 0x6fffffff /* Last of OS specific semantics */
	SHT_LOPROC         SectionType = 0x70000000 /* reserved range for processor */
	SHT_MIPS_ABIFLAGS  SectionType = 0x7000002a /* .MIPS.abiflags */
	SHT_HIPROC         SectionType = 0x7fffffff /* specific section header types */
	SHT_LOUSER         SectionType = 0x80000000 /* reserved range for application */
	SHT_HIUSER         SectionType = 0xffffffff /* specific indexes */
)

var shtStrings = []intName{
	{0, "SHT_NULL"},
	{1, "SHT_PROGBITS"},
	{2, "SHT_SYMTAB"},
	{3, "SHT_STRTAB"},
	{4, "SHT_RELA"},
	{5, "SHT_HASH"},
	{6, "SHT_DYNAMIC"},
	{7, "SHT_NOTE"},
	{8, "SHT_NOBITS"},
	{9, "SHT_REL"},
	{10, "SHT_SHLIB"},
	{11, "SHT_DYNSYM"},
	{14, "SHT_INIT_ARRAY"},
	{15, "SHT_FINI_ARRAY"},
	{16, "SHT_PREINIT_ARRAY"},
	{17, "SHT_GROUP"},
	{18, "SHT_SYMTAB_SHNDX"},
	{0x60000000, "SHT_LOOS"},
	{0x6ffffff5, "SHT_GNU_ATTRIBUTES"},
	{0x6ffffff6, "SHT_GNU_HASH"},
	{0x6ffffff7, "SHT_GNU_LIBLIST"},
	{0x6ffffffd, "SHT_GNU_VERDEF"},
	{0x6ffffffe, "SHT_GNU_VERNEED"},
	{0x6fffffff, "SHT_GNU_VERSYM"},
	{0x70000000, "SHT_LOPROC"},
	{0x7000002a, "SHT_MIPS_ABIFLAGS"},
	{0x7fffffff, "SHT_HIPROC"},
	{0x80000000, "SHT_LOUSER"},
	{0xffffffff, "SHT_HIUSER"},
}

func (i SectionType) String() string   { return stringName(uint32(i), shtStrings, false) }
func (i SectionType) GoString() string { return stringName(uint32(i), shtStrings, true) }

// Section flags.
type SectionFlag uint32

const (
	SHF_WRITE            SectionFlag = 0x1        /* Section contains writable data. */
	SHF_ALLOC            SectionFlag = 0x2        /* Section occupies memory. */
	SHF_EXECINSTR        SectionFlag = 0x4        /* Section contains instructions. */
	SHF_MERGE            SectionFlag = 0x10       /* Section may be merged. */
	SHF_STRINGS          SectionFlag = 0x20       /* Section contains strings. */
	SHF_INFO_LINK        SectionFlag = 0x40       /* sh_info holds section index. */
	SHF_LINK_ORDER       SectionFlag = 0x80       /* Special ordering requirements. */
	SHF_OS_NONCONFORMING SectionFlag = 0x100      /* OS-specific processing required. */
	SHF_GROUP            SectionFlag = 0x200      /* Member of section group. */
	SHF_TLS              SectionFlag = 0x400      /* Section contains TLS data. */
	SHF_COMPRESSED       SectionFlag = 0x800      /* Section is compressed. */
	SHF_MASKOS           SectionFlag = 0x0ff00000 /* OS-specific semantics. */
	SHF_MASKPROC         SectionFlag = 0xf0000000 /* Processor-specific semantics. */
)

var shfStrings = []intName{
	{0x1, "SHF_WRITE"},
	{0x2, "SHF_ALLOC"},
	{0x4, "SHF_EXECINSTR"},
	{0x10, "SHF_MERGE"},
	{0x20, "SHF_STRINGS"},
	{0x40, "SHF_INFO_LINK"},
	{0x80, "SHF_LINK_ORDER"},
	{0x100, "SHF_OS_NONCONFORMING"},
	{0x200, "SHF_GROUP"},
	{0x400, "SHF_TLS"},
	{0x800, "SHF_COMPRESSED"},
}

func (i SectionFlag) String() string   { return flagName(uint32(i), shfStrings, false) }
func (i SectionFlag) GoString() string { return flagName(uint32(i), shfStrings, true) }

// Section compression type.
type CompressionType int

const (
	COMPRESS_ZLIB   CompressionType = 1          /* ZLIB compression. */
	COMPRESS_ZSTD   CompressionType = 2          /* ZSTD compression. */
	COMPRESS_LOOS   CompressionType = 0x60000000 /* First OS-specific. */
	COMPRESS_HIOS   CompressionType = 0x6fffffff /* Last OS-specific. */
	COMPRESS_LOPROC CompressionType = 0x70000000 /* First processor-specific type. */
	COMPRESS_HIPROC CompressionType = 0x7fffffff /* Last processor-specific type. */
)

var compressionStrings = []intName{
	{1, "COMPRESS_ZLIB"},
	{2, "COMPRESS_ZSTD"},
	{0x60000000, "COMPRESS_LOOS"},
	{0x6fffffff, "COMPRESS_HIOS"},
	{0x70000000, "COMPRESS_LOPROC"},
	{0x7fffffff, "COMPRESS_HIPROC"},
}

func (i CompressionType) String() string   { return stringName(uint32(i), compressionStrings, false) }
func (i CompressionType) GoString() string { return stringName(uint32(i), compressionStrings, true) }

// Prog.Type
type ProgType int

const (
	PT_NULL    ProgType = 0 /* Unused entry. */
	PT_LOAD    ProgType = 1 /* Loadable segment. */
	PT_DYNAMIC ProgType = 2 /* Dynamic linking information segment. */
	PT_INTERP  ProgType = 3 /* Pathname of interpreter. */
	PT_NOTE    ProgType = 4 /* Auxiliary information. */
	PT_SHLIB   ProgType = 5 /* Reserved (not used). */
	PT_PHDR    ProgType = 6 /* Location of program header itself. */
	PT_TLS     ProgType = 7 /* Thread local storage segment */

	PT_LOOS ProgType = 0x60000000 /* First OS-specific. */

	PT_GNU_EH_FRAME ProgType = 0x6474e550 /* Frame unwind information */
	PT_GNU_STACK    ProgType = 0x6474e551 /* Stack flags */
	PT_GNU_RELRO    ProgType = 0x6474e552 /* Read only after relocs */
	PT_GNU_PROPERTY ProgType = 0x6474e553 /* GNU property */
	PT_GNU_MBIND_LO ProgType = 0x6474e555 /* Mbind segments start */
	PT_GNU_MBIND_HI ProgType = 0x6474f554 /* Mbind segments finish */

	PT_PAX_FLAGS ProgType = 0x65041580 /* PAX flags */

	PT_OPENBSD_RANDOMIZE ProgType = 0x65a3dbe6 /* Random data */
	PT_OPENBSD_WXNEEDED  ProgType = 0x65a3dbe7 /* W^X violations */
	PT_OPENBSD_BOOTDATA  ProgType = 0x65a41be6 /* Boot arguments */

	PT_SUNW_EH_FRAME ProgType = 0x6474e550 /* Frame unwind information */
	PT_SUNWSTACK     ProgType = 0x6ffffffb /* Stack segment */

	PT_HIOS ProgType = 0x6fffffff /* Last OS-specific. */

	PT_LOPROC ProgType = 0x70000000 /* First processor-specific type. */

	PT_ARM_ARCHEXT ProgType = 0x70000000 /* Architecture compatibility */
	PT_ARM_EXIDX   ProgType = 0x70000001 /* Exception unwind tables */

	PT_AARCH64_ARCHEXT ProgType = 0x70000000 /* Architecture compatibility */
	PT_AARCH64_UNWIND  ProgType = 0x70000001 /* Exception unwind tables */

	PT_MIPS_REGINFO  ProgType = 0x70000000 /* Register usage */
	PT_MIPS_RTPROC   ProgType = 0x70000001 /* Runtime procedures */
	PT_MIPS_OPTIONS  ProgType = 0x70000002 /* Options */
	PT_MIPS_ABIFLAGS ProgType = 0x70000003 /* ABI flags */

	PT_S390_PGSTE ProgType = 0x70000000 /* 4k page table size */

	PT_HIPROC ProgType = 0x7fffffff /* Last processor-specific type. */
)

var ptStrings = []intName{
	{0, "PT_NULL"},
	{1, "PT_LOAD"},
	{2, "PT_DYNAMIC"},
	{3, "PT_INTERP"},
	{4, "PT_NOTE"},
	{5, "PT_SHLIB"},
	{6, "PT_PHDR"},
	{7, "PT_TLS"},
	{0x60000000, "PT_LOOS"},
	{0x6474e550, "PT_GNU_EH_FRAME"},
	{0x6474e551, "PT_GNU_STACK"},
	{0x6474e552, "PT_GNU_RELRO"},
	{0x6474e553, "PT_GNU_PROPERTY"},
	{0x65041580, "PT_PAX_FLAGS"},
	{0x65a3dbe6, "PT_OPENBSD_RANDOMIZE"},
	{0x65a3dbe7, "PT_OPENBSD_WXNEEDED"},
	{0x65a41be6, "PT_OPENBSD_BOOTDATA"},
	{0x6ffffffb, "PT_SUNWSTACK"},
	{0x6fffffff, "PT_HIOS"},
	{0x70000000, "PT_LOPROC"},
	// We don't list the processor-dependent ProgTypes,
	// as the values overlap.
	{0x7fffffff, "PT_HIPROC"},
}

func (i ProgType) String() string   { return stringName(uint32(i), ptStrings, false) }
func (i ProgType) GoString() string { return stringName(uint32(i), ptStrings, true) }

// Prog.Flag
type ProgFlag uint32

const (
	PF_X        ProgFlag = 0x1        /* Executable. */
	PF_W        ProgFlag = 0x2        /* Writable. */
	PF_R        ProgFlag = 0x4        /* Readable. */
	PF_MASKOS   ProgFlag = 0x0ff00000 /* Operating system-specific. */
	PF_MASKPROC ProgFlag = 0xf0000000 /* Processor-specific. */
)

var pfStrings = []intName{
	{0x1, "PF_X"},
	{0x2, "PF_W"},
	{0x4, "PF_R"},
}

func (i ProgFlag) String() string   { return flagName(uint32(i), pfStrings, false) }
func (i ProgFlag) GoString() string { return flagName(uint32(i), pfStrings, true) }

// Dyn.Tag
type DynTag int

const (
	DT_NULL         DynTag = 0  /* Terminating entry. */
	DT_NEEDED       DynTag = 1  /* String table offset of a needed shared library. */
	DT_PLTRELSZ     DynTag = 2  /* Total size in bytes of PLT relocations. */
	DT_PLTGOT       DynTag = 3  /* Processor-dependent address. */
	DT_HASH         DynTag = 4  /* Address of symbol hash table. */
	DT_STRTAB       DynTag = 5  /* Address of string table. */
	DT_SYMTAB       DynTag = 6  /* Address of symbol table. */
	DT_RELA         DynTag = 7  /* Address of ElfNN_Rela relocations. */
	DT_RELASZ       DynTag = 8  /* Total size of ElfNN_Rela relocations. */
	DT_RELAENT      DynTag = 9  /* Size of each ElfNN_Rela relocation entry. */
	DT_STRSZ        DynTag = 10 /* Size of string table. */
	DT_SYMENT       DynTag = 11 /* Size of each symbol table entry. */
	DT_INIT         DynTag = 12 /* Address of initialization function. */
	DT_FINI         DynTag = 13 /* Address of finalization function. */
	DT_SONAME       DynTag = 14 /* String table offset of shared object name. */
	DT_RPATH        DynTag = 15 /* String table offset of library path. [sup] */
	DT_SYMBOLIC     DynTag = 16 /* Indicates "symbolic" linking. [sup] */
	DT_REL          DynTag = 17 /* Address of ElfNN_Rel relocations. */
	DT_RELSZ        DynTag = 18 /* Total size of ElfNN_Rel relocations. */
	DT_RELENT       DynTag = 19 /* Size of each ElfNN_Rel relocation. */
	DT_PLTREL       DynTag = 20 /* Type of relocation used for PLT. */
	DT_DEBUG        DynTag = 21 /* Reserved (not used). */
	DT_TEXTREL      DynTag = 22 /* Indicates there may be relocations in non-writable segments. [sup] */
	DT_JMPREL       DynTag = 23 /* Address of PLT relocations. */
	DT_BIND_NOW     DynTag = 24 /* [sup] */
	DT_INIT_ARRAY   DynTag = 25 /* Address of the array of pointers to initialization functions */
	DT_FINI_ARRAY   DynTag = 26 /* Address of the array of pointers to termination functions */
	DT_INIT_ARRAYSZ DynTag = 27 /* Size in bytes of the array of initialization functions. */
	DT_FINI_ARRAYSZ DynTag = 28 /* Size in bytes of the array of termination functions. */
	DT_RUNPATH      DynTag = 29 /* String table offset of a null-terminated library search path string. */
	DT_FLAGS        DynTag = 30 /* Object specific flag values. */
	DT_ENCODING     DynTag = 32 /* Values greater than or equal to DT_ENCODING
	   and less than DT_LOOS follow the rules for
	   the interpretation of the d_un union
	   as follows: even == 'd_ptr', even == 'd_val'
	   or none */
	DT_PREINIT_ARRAY   DynTag = 32 /* Address of the array of pointers to pre-initialization functions. */
	DT_PREINIT_ARRAYSZ DynTag = 33 /* Size in bytes of the array of pre-initialization functions. */
	DT_SYMTAB_SHNDX    DynTag = 34 /* Address of SHT_SYMTAB_SHNDX section. */

	DT_LOOS DynTag = 0x6000000d /* First OS-specific */
	DT_HIOS DynTag = 0x6ffff000 /* Last OS-specific */

	DT_VALRNGLO       DynTag = 0x6ffffd00
	DT_GNU_PRELINKED  DynTag = 0x6ffffdf5
	DT_GNU_CONFLICTSZ DynTag = 0x6ffffdf6
	DT_GNU_LIBLISTSZ  DynTag = 0x6ffffdf7
	DT_CHECKSUM       DynTag = 0x6ffffdf8
	DT_PLTPADSZ       DynTag = 0x6ffffdf9
	DT_MOVEENT        DynTag = 0x6ffffdfa
	DT_MOVESZ         DynTag = 0x6ffffdfb
	DT_FEATURE        DynTag = 0x6ffffdfc
	DT_POSFLAG_1      DynTag = 0x6ffffdfd
	DT_SYMINSZ        DynTag = 0x6ffffdfe
	DT_SYMINENT       DynTag = 0x6ffffdff
	DT_VALRNGHI       DynTag = 0x6ffffdff

	DT_ADDRRNGLO    DynTag = 0x6ffffe00
	DT_GNU_HASH     DynTag = 0x6ffffef5
	DT_TLSDESC_PLT  DynTag = 0x6ffffef6
	DT_TLSDESC_GOT  DynTag = 0x6ffffef7
	DT_GNU_CONFLICT DynTag = 0x6ffffef8
	DT_GNU_LIBLIST  DynTag = 0x6ffffef9
	DT_CONFIG       DynTag = 0x6ffffefa
	DT_DEPAUDIT     DynTag = 0x6ffffefb
	DT_AUDIT        DynTag = 0x6ffffefc
	DT_PLTPAD       DynTag = 0x6ffffefd
	DT_MOVETAB      DynTag = 0x6ffffefe
	DT_SYMINFO      DynTag = 0x6ffffeff
	DT_ADDRRNGHI    DynTag = 0x6ffffeff

	DT_VERSYM     DynTag = 0x6ffffff0
	DT_RELACOUNT  DynTag = 0x6ffffff9
	DT_RELCOUNT   DynTag = 0x6ffffffa
	DT_FLAGS_1    DynTag = 0x6ffffffb
	DT_VERDEF     DynTag = 0x6ffffffc
	DT_VERDEFNUM  DynTag = 0x6ffffffd
	DT_VERNEED    DynTag = 0x6ffffffe
	DT_VERNEEDNUM DynTag = 0x6fffffff

	DT_LOPROC DynTag = 0x70000000 /* First processor-specific type. */

	DT_MIPS_RLD_VERSION           DynTag = 0x70000001
	DT_MIPS_TIME_STAMP            DynTag = 0x70000002
	DT_MIPS_ICHECKSUM             DynTag = 0x70000003
	DT_MIPS_IVERSION              DynTag = 0x70000004
	DT_MIPS_FLAGS                 DynTag = 0x70000005
	DT_MIPS_BASE_ADDRESS          DynTag = 0x70000006
	DT_MIPS_MSYM                  DynTag = 0x70000007
	DT_MIPS_CONFLICT              DynTag = 0x70000008
	DT_MIPS_LIBLIST               DynTag = 0x70000009
	DT_MIPS_LOCAL_GOTNO           DynTag = 0x7000000a
	DT_MIPS_CONFLICTNO            DynTag = 0x7000000b
	DT_MIPS_LIBLISTNO             DynTag = 0x70000010
	DT_MIPS_SYMTABNO              DynTag = 0x70000011
	DT_MIPS_UNREFEXTNO            DynTag = 0x70000012
	DT_MIPS_GOTSYM                DynTag = 0x70000013
	DT_MIPS_HIPAGENO              DynTag = 0x70000014
	DT_MIPS_RLD_MAP               DynTag = 0x70000016
	DT_MIPS_DELTA_CLASS           DynTag = 0x70000017
	DT_MIPS_DELTA_CLASS_NO        DynTag = 0x70000018
	DT_MIPS_DELTA_INSTANCE        DynTag = 0x70000019
	DT_MIPS_DELTA_INSTANCE_NO     DynTag = 0x7000001a
	DT_MIPS_DELTA_RELOC           DynTag = 0x7000001b
	DT_MIPS_DELTA_RELOC_NO        DynTag = 0x7000001c
	DT_MIPS_DELTA_SYM             DynTag = 0x7000001d
	DT_MIPS_DELTA_SYM_NO          DynTag = 0x7000001e
	DT_MIPS_DELTA_CLASSSYM        DynTag = 0x70000020
	DT_MIPS_DELTA_CLASSSYM_NO     DynTag = 0x70000021
	DT_MIPS_CXX_FLAGS             DynTag = 0x70000022
	DT_MIPS_PIXIE_INIT            DynTag = 0x70000023
	DT_MIPS_SYMBOL_LIB            DynTag = 0x70000024
	DT_MIPS_LOCALPAGE_GOTIDX      DynTag = 0x70000025
	DT_MIPS_LOCAL_GOTIDX          DynTag = 0x70000026
	DT_MIPS_HIDDEN_GOTIDX         DynTag = 0x70000027
	DT_MIPS_PROTECTED_GOTIDX      DynTag = 0x70000028
	DT_MIPS_OPTIONS               DynTag = 0x70000029
	DT_MIPS_INTERFACE             DynTag = 0x7000002a
	DT_MIPS_DYNSTR_ALIGN          DynTag = 0x7000002b
	DT_MIPS_INTERFACE_SIZE        DynTag = 0x7000002c
	DT_MIPS_RLD_TEXT_RESOLVE_ADDR DynTag = 0x7000002d
	DT_MIPS_PERF_SUFFIX           DynTag = 0x7000002e
	DT_MIPS_COMPACT_SIZE          DynTag = 0x7000002f
	DT_MIPS_GP_VALUE              DynTag = 0x70000030
	DT_MIPS_AUX_DYNAMIC           DynTag = 0x70000031
	DT_MIPS_PLTGOT                DynTag = 0x70000032
	DT_MIPS_RWPLT                 DynTag = 0x70000034
	DT_MIPS_RLD_MAP_REL           DynTag = 0x70000035

	DT_PPC_GOT DynTag = 0x70000000
	DT_PPC_OPT DynTag = 0x70000001

	DT_PPC64_GLINK DynTag = 0x70000000
	DT_PPC64_OPD   DynTag = 0x70000001
	DT_PPC64_OPDSZ DynTag = 0x70000002
	DT_PPC64_OPT   DynTag = 0x70000003

	DT_SPARC_REGISTER DynTag = 0x70000001

	DT_AUXILIARY DynTag = 0x7ffffffd
	DT_USED      DynTag = 0x7ffffffe
	DT_FILTER    DynTag = 0x7fffffff

	DT_HIPROC DynTag = 0x7fffffff /* Last processor-specific type. */
)

var dtStrings = []intName{
	{0, "DT_NULL"},
	{1, "DT_NEEDED"},
	{2, "DT_PLTRELSZ"},
	{3, "DT_PLTGOT"},
	{4, "DT_HASH"},
	{5, "DT_STRTAB"},
	{6, "DT_SYMTAB"},
	{7, "DT_RELA"},
	{8, "DT_RELASZ"},
	{9, "DT_RELAENT"},
	{10, "DT_STRSZ"},
	{11, "DT_SYMENT"},
	{12, "DT_INIT"},
	{13, "DT_FINI"},
	{14, "DT_SONAME"},
	{15, "DT_RPATH"},
	{16, "DT_SYMBOLIC"},
	{17, "DT_REL"},
	{18, "DT_RELSZ"},
	{19, "DT_RELENT"},
	{20, "DT_PLTREL"},
	{21, "DT_DEBUG"},
	{22, "DT_TEXTREL"},
	{23, "DT_JMPREL"},
	{24, "DT_BIND_NOW"},
	{25, "DT_INIT_ARRAY"},
	{26, "DT_FINI_ARRAY"},
	{27, "DT_INIT_ARRAYSZ"},
	{28, "DT_FINI_ARRAYSZ"},
	{29, "DT_RUNPATH"},
	{30, "DT_FLAGS"},
	{32, "DT_ENCODING"},
	{32, "DT_PREINIT_ARRAY"},
	{33, "DT_PREINIT_ARRAYSZ"},
	{34, "DT_SYMTAB_SHNDX"},
	{0x6000000d, "DT_LOOS"},
	{0x6ffff000, "DT_HIOS"},
	{0x6ffffd00, "DT_VALRNGLO"},
	{0x6ffffdf5, "DT_GNU_PRELINKED"},
	{0x6ffffdf6, "DT_GNU_CONFLICTSZ"},
	{0x6ffffdf7, "DT_GNU_LIBLISTSZ"},
	{0x6ffffdf8, "DT_CHECKSUM"},
	{0x6ffffdf9, "DT_PLTPADSZ"},
	{0x6ffffdfa, "DT_MOVEENT"},
	{0x6ffffdfb, "DT_MOVESZ"},
	{0x6ffffdfc, "DT_FEATURE"},
	{0x6ffffdfd, "DT_POSFLAG_1"},
	{0x6ffffdfe, "DT_SYMINSZ"},
	{0x6ffffdff, "DT_SYMINENT"},
	{0x6ffffdff, "DT_VALRNGHI"},
	{0x6ffffe00, "DT_ADDRRNGLO"},
	{0x6ffffef5, "DT_GNU_HASH"},
	{0x6ffffef6, "DT_TLSDESC_PLT"},
	{0x6ffffef7, "DT_TLSDESC_GOT"},
	{0x6ffffef8, "DT_GNU_CONFLICT"},
	{0x6ffffef9, "DT_GNU_LIBLIST"},
	{0x6ffffefa, "DT_CONFIG"},
	{0x6ffffefb, "DT_DEPAUDIT"},
	{0x6ffffefc, "DT_AUDIT"},
	{0x6ffffefd, "DT_PLTPAD"},
	{0x6ffffefe, "DT_MOVETAB"},
	{0x6ffffeff, "DT_SYMINFO"},
	{0x6ffffeff, "DT_ADDRRNGHI"},
	{0x6ffffff0, "DT_VERSYM"},
	{0x6ffffff9, "DT_RELACOUNT"},
	{0x6ffffffa, "DT_RELCOUNT"},
	{0x6ffffffb, "DT_FLAGS_1"},
	{0x6ffffffc, "DT_VERDEF"},
	{0x6ffffffd, "DT_VERDEFNUM"},
	{0x6ffffffe, "DT_VERNEED"},
	{0x6fffffff, "DT_VERNEEDNUM"},
	{0x70000000, "DT_LOPROC"},
	// We don't list the processor-dependent DynTags,
	// as the values overlap.
	{0x7ffffffd, "DT_AUXILIARY"},
	{0x7ffffffe, "DT_USED"},
	{0x7fffffff, "DT_FILTER"},
}

func (i DynTag) String() string   { return stringName(uint32(i), dtStrings, false) }
func (i DynTag) GoString() string { return stringName(uint32(i), dtStrings, true) }

// DT_FLAGS values.
type DynFlag int

const (
	DF_ORIGIN DynFlag = 0x0001 /* Indicates that the object being loaded may
	   make reference to the
	   $ORIGIN substitution string */
	DF_SYMBOLIC DynFlag = 0x0002 /* Indicates "symbolic" linking. */
	DF_TEXTREL  DynFlag = 0x0004 /* Indicates there may be relocations in non-writable segments. */
	DF_BIND_NOW DynFlag = 0x0008 /* Indicates that the dynamic linker should
	   process all relocations for the object
	   containing this entry before transferring
	   control to the program. */
	DF_STATIC_TLS DynFlag = 0x0010 /* Indicates that the shared object or
	   executable contains code using a static
	   thread-local storage scheme. */
)

var dflagStrings = []intName{
	{0x0001, "DF_ORIGIN"},
	{0x0002, "DF_SYMBOLIC"},
	{0x0004, "DF_TEXTREL"},
	{0x0008, "DF_BIND_NOW"},
	{0x0010, "DF_STATIC_TLS"},
}

func (i DynFlag) String() string   { return flagName(uint32(i), dflagStrings, false) }
func (i DynFlag) GoString() string { return flagName(uint32(i), dflagStrings, true) }

// DT_FLAGS_1 values.
type DynFlag1 uint32

const (
	// Indicates that all relocations for this object must be processed before
	// returning control to the program.
	DF_1_NOW DynFlag1 = 0x00000001
	// Unused.
	DF_1_GLOBAL DynFlag1 = 0x00000002
	// Indicates that the object is a member of a group.
	DF_1_GROUP DynFlag1 = 0x00000004
	// Indicates that the object cannot be deleted from a process.
	DF_1_NODELETE DynFlag1 = 0x00000008
	// Meaningful only for filters. Indicates that all associated filtees be
	// processed immediately.
	DF_1_LOADFLTR DynFlag1 = 0x00000010
	// Indicates that this object's initialization section be run before any other
	// objects loaded.
	DF_1_INITFIRST DynFlag1 = 0x00000020
	// Indicates that the object cannot be added to a running process with dlopen.
	DF_1_NOOPEN DynFlag1 = 0x00000040
	// Indicates the object requires $ORIGIN processing.
	DF_1_ORIGIN DynFlag1 = 0x00000080
	// Indicates that the object should use direct binding information.
	DF_1_DIRECT DynFlag1 = 0x00000100
	// Unused.
	DF_1_TRANS DynFlag1 = 0x00000200
	// Indicates that the objects symbol table is to interpose before all symbols
	// except the primary load object, which is typically the executable.
	DF_1_INTERPOSE DynFlag1 = 0x00000400
	// Indicates that the search for dependencies of this object ignores any
	// default library search paths.
	DF_1_NODEFLIB DynFlag1 = 0x00000800
	// Indicates that this object is not dumped by dldump. Candidates are objects
	// with no relocations that might get included when generating alternative
	// objects using.
	DF_1_NODUMP DynFlag1 = 0x00001000
	// Identifies this object as a configuration alternative object generated by
	// crle. Triggers the runtime linker to search for a configuration file $ORIGIN/ld.config.app-name.
	DF_1_CONFALT DynFlag1 = 0x00002000
	// Meaningful only for filtees. Terminates a filters search for any
	// further filtees.
	DF_1_ENDFILTEE DynFlag1 = 0x00004000
	// Indicates that this object has displacement relocations applied.
	DF_1_DISPRELDNE DynFlag1 = 0x00008000
	// Indicates that this object has displacement relocations pending.
	DF_1_DISPRELPND DynFlag1 = 0x00010000
	// Indicates that this object contains symbols that cannot be directly
	// bound to.
	DF_1_NODIRECT DynFlag1 = 0x00020000
	// Reserved for internal use by the kernel runtime-linker.
	DF_1_IGNMULDEF DynFlag1 = 0x00040000
	// Reserved for internal use by the kernel runtime-linker.
	DF_1_NOKSYMS DynFlag1 = 0x00080000
	// Reserved for internal use by the kernel runtime-linker.
	DF_1_NOHDR DynFlag1 = 0x00100000
	// Indicates that this object has been edited or has been modified since the
	// objects original construction by the link-editor.
	DF_1_EDITED DynFlag1 = 0x00200000
	// Reserved for internal use by the kernel runtime-linker.
	DF_1_NORELOC DynFlag1 = 0x00400000
	// Indicates that the object contains individual symbols that should interpose
	// before all symbols except the primary load object, which is typically the
	// executable.
	DF_1_SYMINTPOSE DynFlag1 = 0x00800000
	// Indicates that the executable requires global auditing.
	DF_1_GLOBAUDIT DynFlag1 = 0x01000000
	// Indicates that the object defines, or makes reference to singleton symbols.
	DF_1_SINGLETON DynFlag1 = 0x02000000
	// Indicates that the object is a stub.
	DF_1_STUB DynFlag1 = 0x04000000
	// Indicates that the object is a position-independent executable.
	DF_1_PIE DynFlag1 = 0x08000000
	// Indicates that the object is a kernel module.
	DF_1_KMOD DynFlag1 = 0x10000000
	// Indicates that the object is a weak standard filter.
	DF_1_WEAKFILTER DynFlag1 = 0x20000000
	// Unused.
	DF_1_NOCOMMON DynFlag1 = 0x40000000
)

var dflag1Strings = []intName{
	{0x00000001, "DF_1_NOW"},
	{0x00000002, "DF_1_GLOBAL"},
	{0x00000004, "DF_1_GROUP"},
	{0x00000008, "DF_1_NODELETE"},
	{0x00000010, "DF_1_LOADFLTR"},
	{0x00000020, "DF_1_INITFIRST"},
	{0x00000040, "DF_1_NOOPEN"},
	{0x00000080, "DF_1_ORIGIN"},
	{0x00000100, "DF_1_DIRECT"},
	{0x00000200, "DF_1_TRANS"},
	{0x00000400, "DF_1_INTERPOSE"},
	{0x00000800, "DF_1_NODEFLIB"},
	{0x00001000, "DF_1_NODUMP"},
	{0x00002000, "DF_1_CONFALT"},
	{0x00004000, "DF_1_ENDFILTEE"},
	{0x00008000, "DF_1_DISPRELDNE"},
	{0x00010000, "DF_1_DISPRELPND"},
	{0x00020000, "DF_1_NODIRECT"},
	{0x00040000, "DF_1_IGNMULDEF"},
	{0x00080000, "DF_1_NOKSYMS"},
	{0x00100000, "DF_1_NOHDR"},
	{0x00200000, "DF_1_EDITED"},
	{0x00400000, "DF_1_NORELOC"},
	{0x00800000, "DF_1_SYMINTPOSE"},
	{0x01000000, "DF_1_GLOBAUDIT"},
	{0x02000000, "DF_1_SINGLETON"},
	{0x04000000, "DF_1_STUB"},
	{0x08000000, "DF_1_PIE"},
	{0x10000000, "DF_1_KMOD"},
	{0x20000000, "DF_1_WEAKFILTER"},
	{0x40000000, "DF_1_NOCOMMON"},
}

func (i DynFlag1) String() string   { return flagName(uint32(i), dflag1Strings, false) }
func (i DynFlag1) GoString() string { return flagName(uint32(i), dflag1Strings, true) }

// NType values; used in core files.
type NType int

const (
	NT_PRSTATUS NType = 1 /* Process status. */
	NT_FPREGSET NType = 2 /* Floating point registers. */
	NT_PRPSINFO NType = 3 /* Process state info. */
)

var ntypeStrings = []intName{
	{1, "NT_PRSTATUS"},
	{2, "NT_FPREGSET"},
	{3, "NT_PRPSINFO"},
}

func (i NType) String() string   { return stringName(uint32(i), ntypeStrings, false) }
func (i NType) GoString() string { return stringName(uint32(i), ntypeStrings, true) }

/* Symbol Binding - ELFNN_ST_BIND - st_info */
type SymBind int

const (
	STB_LOCAL  SymBind = 0  /* Local symbol */
	STB_GLOBAL SymBind = 1  /* Global symbol */
	STB_WEAK   SymBind = 2  /* like global - lower precedence */
	STB_LOOS   SymBind = 10 /* Reserved range for operating system */
	STB_HIOS   SymBind = 12 /*   specific semantics. */
	STB_LOPROC SymBind = 13 /* reserved range for processor */
	STB_HIPROC SymBind = 15 /*   specific semantics. */
)

var stbStrings = []intName{
	{0, "STB_LOCAL"},
	{1, "STB_GLOBAL"},
	{2, "STB_WEAK"},
	{10, "STB_LOOS"},
	{12, "STB_HIOS"},
	{13, "STB_LOPROC"},
	{15, "STB_HIPROC"},
}

func (i SymBind) String() string   { return stringName(uint32(i), stbStrings, false) }
func (i SymBind) GoString() string { return stringName(uint32(i), stbStrings, true) }

/* Symbol type - ELFNN_ST_TYPE - st_info */
type SymType int

const (
	STT_NOTYPE  SymType = 0  /* Unspecified type. */
	STT_OBJECT  SymType = 1  /* Data object. */
	STT_FUNC    SymType = 2  /* Function. */
	STT_SECTION SymType = 3  /* Section. */
	STT_FILE    SymType = 4  /* Source file. */
	STT_COMMON  SymType = 5  /* Uninitialized common block. */
	STT_TLS     SymType = 6  /* TLS object. */
	STT_LOOS    SymType = 10 /* Reserved range for operating system */
	STT_HIOS    SymType = 12 /*   specific semantics. */
	STT_LOPROC  SymType = 13 /* reserved range for processor */
	STT_HIPROC  SymType = 15 /*   specific semantics. */
)

var sttStrings = []intName{
	{0, "STT_NOTYPE"},
	{1, "STT_OBJECT"},
	{2, "STT_FUNC"},
	{3, "STT_SECTION"},
	{4, "STT_FILE"},
	{5, "STT_COMMON"},
	{6, "STT_TLS"},
	{10, "STT_LOOS"},
	{12, "STT_HIOS"},
	{13, "STT_LOPROC"},
	{15, "STT_HIPROC"},
}

func (i SymType) String() string   { return stringName(uint32(i), sttStrings, false) }
func (i SymType) GoString() string { return stringName(uint32(i), sttStrings, true) }

/* Symbol visibility - ELFNN_ST_VISIBILITY - st_other */
type SymVis int

const (
	STV_DEFAULT   SymVis = 0x0 /* Default visibility (see binding). */
	STV_INTERNAL  SymVis = 0x1 /* Special meaning in relocatable objects. */
	STV_HIDDEN    SymVis = 0x2 /* Not visible. */
	STV_PROTECTED SymVis = 0x3 /* Visible but not preemptible. */
)

var stvStrings = []intName{
	{0x0, "STV_DEFAULT"},
	{0x1, "STV_INTERNAL"},
	{0x2, "STV_HIDDEN"},
	{0x3, "STV_PROTECTED"},
}

func (i SymVis) String() string   { return stringName(uint32(i), stvStrings, false) }
func (i SymVis) GoString() string { return stringName(uint32(i), stvStrings, true) }

/*
 * Relocation types.
 */

// Relocation types for x86-64.
type R_X86_64 int

const (
	R_X86_64_NONE            R_X86_64 = 0  /* No relocation. */
	R_X86_64_64              R_X86_64 = 1  /* Add 64 bit symbol value. */
	R_X86_64_PC32            R_X86_64 = 2  /* PC-relative 32 bit signed sym value. */
	R_X86_64_GOT32           R_X86_64 = 3  /* PC-relative 32 bit GOT offset. */
	R_X86_64_PLT32           R_X86_64 = 4  /* PC-relative 32 bit PLT offset. */
	R_X86_64_COPY            R_X86_64 = 5  /* Copy data from shared object. */
	R_X86_64_GLOB_DAT        R_X86_64 = 6  /* Set GOT entry to data address. */
	R_X86_64_JMP_SLOT        R_X86_64 = 7  /* Set GOT entry to code address. */
	R_X86_64_RELATIVE        R_X86_64 = 8  /* Add load address of shared object. */
	R_X86_64_GOTPCREL        R_X86_64 = 9  /* Add 32 bit signed pcrel offset to GOT. */
	R_X86_64_32              R_X86_64 = 10 /* Add 32 bit zero extended symbol value */
	R_X86_64_32S             R_X86_64 = 11 /* Add 32 bit sign extended symbol value */
	R_X86_64_16              R_X86_64 = 12 /* Add 16 bit zero extended symbol value */
	R_X86_64_PC16            R_X86_64 = 13 /* Add 16 bit signed extended pc relative symbol value */
	R_X86_64_8               R_X86_64 = 14 /* Add 8 bit zero extended symbol value */
	R_X86_64_PC8             R_X86_64 = 15 /* Add 8 bit signed extended pc relative symbol value */
	R_X86_64_DTPMOD64        R_X86_64 = 16 /* ID of module containing symbol */
	R_X86_64_DTPOFF64        R_X86_64 = 17 /* Offset in TLS block */
	R_X86_64_TPOFF64         R_X86_64 = 18 /* Offset in static TLS block */
	R_X86_64_TLSGD           R_X86_64 = 19 /* PC relative offset to GD GOT entry */
	R_X86_64_TLSLD           R_X86_64 = 20 /* PC relative offset to LD GOT entry */
	R_X86_64_DTPOFF32        R_X86_64 = 21 /* Offset in TLS block */
	R_X86_64_GOTTPOFF        R_X86_64 = 22 /* PC relative offset to IE GOT entry */
	R_X86_64_TPOFF32         R_X86_64 = 23 /* Offset in static TLS block */
	R_X86_64_PC64            R_X86_64 = 24 /* PC relative 64-bit sign extended symbol value. */
	R_X86_64_GOTOFF64        R_X86_64 = 25
	R_X86_64_GOTPC32         R_X86_64 = 26
	R_X86_64_GOT64           R_X86_64 = 27
	R_X86_64_GOTPCREL64      R_X86_64 = 28
	R_X86_64_GOTPC64         R_X86_64 = 29
	R_X86_64_GOTPLT64        R_X86_64 = 30
	R_X86_64_PLTOFF64        R_X86_64 = 31
	R_X86_64_SIZE32          R_X86_64 = 32
	R_X86_64_SIZE64          R_X86_64 = 33
	R_X86_64_GOTPC32_TLSDESC R_X86_64 = 34
	R_X86_64_TLSDESC_CALL    R_X86_64 = 35
	R_X86_64_TLSDESC         R_X86_64 = 36
	R_X86_64_IRELATIVE       R_X86_64 = 37
	R_X86_64_RELATIVE64      R_X86_64 = 38
	R_X86_64_PC32_BND        R_X86_64 = 39
	R_X86_64_PLT32_BND       R_X86_64 = 40
	R_X86_64_GOTPCRELX       R_X86_64 = 41
	R_X86_64_REX_GOTPCRELX   R_X86_64 = 42
)

var rx86_64Strings = []intName{
	{0, "R_X86_64_NONE"},
	{1, "R_X86_64_64"},
	{2, "R_X86_64_PC32"},
	{3, "R_X86_64_GOT32"},
	{4, "R_X86_64_PLT32"},
	{5, "R_X86_64_COPY"},
	{6, "R_X86_64_GLOB_DAT"},
	{7, "R_X86_64_JMP_SLOT"},
	{8, "R_X86_64_RELATIVE"},
	{9, "R_X86_64_GOTPCREL"},
	{10, "R_X86_64_32"},
	{11, "R_X86_64_32S"},
	{12, "R_X86_64_16"},
	{13, "R_X86_64_PC16"},
	{14, "R_X86_64_8"},
	{15, "R_X86_64_PC8"},
	{16, "R_X86_64_DTPMOD64"},
	{17, "R_X86_64_DTPOFF64"},
	{18, "R_X86_64_TPOFF64"},
	{19, "R_X86_64_TLSGD"},
	{20, "R_X86_64_TLSLD"},
	{21, "R_X86_64_DTPOFF32"},
	{22, "R_X86_64_GOTTPOFF"},
	{23, "R_X86_64_TPOFF32"},
	{24, "R_X86_64_PC64"},
	{25, "R_X86_64_GOTOFF64"},
	{26, "R_X86_64_GOTPC32"},
	{27, "R_X86_64_GOT64"},
	{28, "R_X86_64_GOTPCREL64"},
	{29, "R_X86_64_GOTPC64"},
	{30, "R_X86_64_GOTPLT64"},
	{31, "R_X86_64_PLTOFF64"},
	{32, "R_X86_64_SIZE32"},
	{33, "R_X86_64_SIZE64"},
	{34, "R_X86_64_GOTPC32_TLSDESC"},
	{35, "R_X86_64_TLSDESC_CALL"},
	{36, "R_X86_64_TLSDESC"},
	{37, "R_X86_64_IRELATIVE"},
	{38, "R_X86_64_RELATIVE64"},
	{39, "R_X86_64_PC32_BND"},
	{40, "R_X86_64_PLT32_BND"},
	{41, "R_X86_64_GOTPCRELX"},
	{42, "R_X86_64_REX_GOTPCRELX"},
}

func (i R_X86_64) String() string   { return stringName(uint32(i), rx86_64Strings, false) }
func (i R_X86_64) GoString() string { return stringName(uint32(i), rx86_64Strings, true) }

// Relocation types for AArch64 (aka arm64)
type R_AARCH64 int

const (
	R_AARCH64_NONE                            R_AARCH64 = 0
	R_AARCH64_P32_ABS32                       R_AARCH64 = 1
	R_AARCH64_P32_ABS16                       R_AARCH64 = 2
	R_AARCH64_P32_PREL32                      R_AARCH64 = 3
	R_AARCH64_P32_PREL16                      R_AARCH64 = 4
	R_AARCH64_P32_MOVW_UABS_G0                R_AARCH64 = 5
	R_AARCH64_P32_MOVW_UABS_G0_NC             R_AARCH64 = 6
	R_AARCH64_P32_MOVW_UABS_G1                R_AARCH64 = 7
	R_AARCH64_P32_MOVW_SABS_G0                R_AARCH64 = 8
	R_AARCH64_P32_LD_PREL_LO19                R_AARCH64 = 9
	R_AARCH64_P32_ADR_PREL_LO21               R_AARCH64 = 10
	R_AARCH64_P32_ADR_PREL_PG_HI21            R_AARCH64 = 11
	R_AARCH64_P32_ADD_ABS_LO12_NC             R_AARCH64 = 12
	R_AARCH64_P32_LDST8_ABS_LO12_NC           R_AARCH64 = 13
	R_AARCH64_P32_LDST16_ABS_LO12_NC          R_AARCH64 = 14
	R_AARCH64_P32_LDST32_ABS_LO12_NC          R_AARCH64 = 15
	R_AARCH64_P32_LDST64_ABS_LO12_NC          R_AARCH64 = 16
	R_AARCH64_P32_LDST128_ABS_LO12_NC         R_AARCH64 = 17
	R_AARCH64_P32_TSTBR14                     R_AARCH64 = 18
	R_AARCH64_P32_CONDBR19                    R_AARCH64 = 19
	R_AARCH64_P32_JUMP26                      R_AARCH64 = 20
	R_AARCH64_P32_CALL26                      R_AARCH64 = 21
	R_AARCH64_P32_GOT_LD_PREL19               R_AARCH64 = 25
	R_AARCH64_P32_ADR_GOT_PAGE                R_AARCH64 = 26
	R_AARCH64_P32_LD32_GOT_LO12_NC            R_AARCH64 = 27
	R_AARCH64_P32_TLSGD_ADR_PAGE21            R_AARCH64 = 81
	R_AARCH64_P32_TLSGD_ADD_LO12_NC           R_AARCH64 = 82
	R_AARCH64_P32_TLSIE_ADR_GOTTPREL_PAGE21   R_AARCH64 = 103
	R_AARCH64_P32_TLSIE_LD32_GOTTPREL_LO12_NC R_AARCH64 = 104
	R_AARCH64_P32_TLSIE_LD_GOTTPREL_PREL19    R_AARCH64 = 105
	R_AARCH64_P32_TLSLE_MOVW_TPREL_G1         R_AARCH64 = 106
	R_AARCH64_P32_TLSLE_MOVW_TPREL_G0         R_AARCH64 = 107
	R_AARCH64_P32_TLSLE_MOVW_TPREL_G0_NC      R_AARCH64 = 108
	R_AARCH64_P32_TLSLE_ADD_TPREL_HI12        R_AARCH64 = 109
	R_AARCH64_P32_TLSLE_ADD_TPREL_LO12        R_AARCH64 = 110
	R_AARCH64_P32_TLSLE_ADD_TPREL_LO12_NC     R_AARCH64 = 111
	R_AARCH64_P32_TLSDESC_LD_PREL19           R_AARCH64 = 122
	R_AARCH64_P32_TLSDESC_ADR_PREL21          R_AARCH64 = 123
	R_AARCH64_P32_TLSDESC_ADR_PAGE21          R_AARCH64 = 124
	R_AARCH64_P32_TLSDESC_LD32_LO12_NC        R_AARCH64 = 125
	R_AARCH64_P32_TLSDESC_ADD_LO12_NC         R_AARCH64 = 126
	R_AARCH64_P32_TLSDESC_CALL                R_AARCH64 = 127
	R_AARCH64_P32_COPY                        R_AARCH64 = 180
	R_AARCH64_P32_GLOB_DAT                    R_AARCH64 = 181
	R_AARCH64_P32_JUMP_SLOT                   R_AARCH64 = 182
	R_AARCH64_P32_RELATIVE                    R_AARCH64 = 183
	R_AARCH64_P32_TLS_DTPMOD                  R_AARCH64 = 184
	R_AARCH64_P32_TLS_DTPREL                  R_AARCH64 = 185
	R_AARCH64_P32_TLS_TPREL                   R_AARCH64 = 186
	R_AARCH64_P32_TLSDESC                     R_AARCH64 = 187
	R_AARCH64_P32_IRELATIVE                   R_AARCH64 = 188
	R_AARCH64_NULL                            R_AARCH64 = 256
	R_AARCH64_ABS64                           R_AARCH64 = 257
	R_AARCH64_ABS32                           R_AARCH64 = 258
	R_AARCH64_ABS16                           R_AARCH64 = 259
	R_AARCH64_PREL64                          R_AARCH64 = 260
	R_AARCH64_PREL32                          R_AARCH64 = 261
	R_AARCH64_PREL16                          R_AARCH64 = 262
	R_AARCH64_MOVW_UABS_G0                    R_AARCH64 = 263
	R_AARCH64_MOVW_UABS_G0_NC                 R_AARCH64 = 264
	R_AARCH64_MOVW_UABS_G1                    R_AARCH64 = 265
	R_AARCH64_MOVW_UABS_G1_NC                 R_AARCH64 = 266
	R_AARCH64_MOVW_UABS_G2                    R_AARCH64 = 267
	R_AARCH64_MOVW_UABS_G2_NC                 R_AARCH64 = 268
	R_AARCH64_MOVW_UABS_G3                    R_AARCH64 = 269
	R_AARCH64_MOVW_SABS_G0                    R_AARCH64 = 270
	R_AARCH64_MOVW_SABS_G1                    R_AARCH64 = 271
	R_AARCH64_MOVW_SABS_G2                    R_AARCH64 = 272
	R_AARCH64_LD_PREL_LO19                    R_AARCH64 = 273
	R_AARCH64_ADR_PREL_LO21                   R_AARCH64 = 274
	R_AARCH64_ADR_PREL_PG_HI21                R_AARCH64 = 275
	R_AARCH64_ADR_PREL_PG_HI21_NC             R_AARCH64 = 276
	R_AARCH64_ADD_ABS_LO12_NC                 R_AARCH64 = 277
	R_AARCH64_LDST8_ABS_LO12_NC               R_AARCH64 = 278
	R_AARCH64_TSTBR14                         R_AARCH64 = 279
	R_AARCH64_CONDBR19                        R_AARCH64 = 280
	R_AARCH64_JUMP26                          R_AARCH64 = 282
	R_AARCH64_CALL26                          R_AARCH64 = 283
	R_AARCH64_LDST16_ABS_LO12_NC              R_AARCH64 = 284
	R_AARCH64_LDST32_ABS_LO12_NC              R_AARCH64 = 285
	R_AARCH64_LDST64_ABS_LO12_NC              R_AARCH64 = 286
	R_AARCH64_LDST128_ABS_LO12_NC             R_AARCH64 = 299
	R_AARCH64_GOT_LD_PREL19                   R_AARCH64 = 309
	R_AARCH64_LD64_GOTOFF_LO15                R_AARCH64 = 310
	R_AARCH64_ADR_GOT_PAGE                    R_AARCH64 = 311
	R_AARCH64_LD64_GOT_LO12_NC                R_AARCH64 = 312
	R_AARCH64_LD64_GOTPAGE_LO15               R_AARCH64 = 313
	R_AARCH64_TLSGD_ADR_PREL21                R_AARCH64 = 512
	R_AARCH64_TLSGD_ADR_PAGE21                R_AARCH64 = 513
	R_AARCH64_TLSGD_ADD_LO12_NC               R_AARCH64 = 514
	R_AARCH64_TLSGD_MOVW_G1                   R_AARCH64 = 515
	R_AARCH64_TLSGD_MOVW_G0_NC                R_AARCH64 = 516
	R_AARCH64_TLSLD_ADR_PREL21                R_AARCH64 = 517
	R_AARCH64_TLSLD_ADR_PAGE21                R_AARCH64 = 518
	R_AARCH64_TLSIE_MOVW_GOTTPREL_G1          R_AARCH64 = 539
	R_AARCH64_TLSIE_MOVW_GOTTPREL_G0_NC       R_AARCH64 = 540
	R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21       R_AARCH64 = 541
	R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC     R_AARCH64 = 542
	R_AARCH64_TLSIE_LD_GOTTPREL_PREL19        R_AARCH64 = 543
	R_AARCH64_TLSLE_MOVW_TPREL_G2             R_AARCH64 = 544
	R_AARCH64_TLSLE_MOVW_TPREL_G1             R_AARCH64 = 545
	R_AARCH64_TLSLE_MOVW_TPREL_G1_NC          R_AARCH64 = 546
	R_AARCH64_TLSLE_MOVW_TPREL_G0             R_AARCH64 = 547
	R_AARCH64_TLSLE_MOVW_TPREL_G0_NC          R_AARCH64 = 548
	R_AARCH64_TLSLE_ADD_TPREL_HI12            R_AARCH64 = 549
	R_AARCH64_TLSLE_ADD_TPREL_LO12            R_AARCH64 = 550
	R_AARCH64_TLSLE_ADD_TPREL_LO12_NC         R_AARCH64 = 551
	R_AARCH64_TLSDESC_LD_PREL19               R_AARCH64 = 560
	R_AARCH64_TLSDESC_ADR_PREL21              R_AARCH64 = 561
	R_AARCH64_TLSDESC_ADR_PAGE21              R_AARCH64 = 562
	R_AARCH64_TLSDESC_LD64_LO12_NC            R_AARCH64 = 563
	R_AARCH64_TLSDESC_ADD_LO12_NC             R_AARCH64 = 564
	R_AARCH64_TLSDESC_OFF_G1                  R_AARCH64 = 565
	R_AARCH64_TLSDESC_OFF_G0_NC               R_AARCH64 = 566
	R_AARCH64_TLSDESC_LDR                     R_AARCH64 = 567
	R_AARCH64_TLSDESC_ADD                     R_AARCH64 = 568
	R_AARCH64_TLSDESC_CALL                    R_AARCH64 = 569
	R_AARCH64_TLSLE_LDST128_TPREL_LO12        R_AARCH64 = 570
	R_AARCH64_TLSLE_LDST128_TPREL_LO12_NC     R_AARCH64 = 571
	R_AARCH64_TLSLD_LDST128_DTPREL_LO12       R_AARCH64 = 572
	R_AARCH64_TLSLD_LDST128_DTPREL_LO12_NC    R_AARCH64 = 573
	R_AARCH64_COPY                            R_AARCH64 = 1024
	R_AARCH64_GLOB_DAT                        R_AARCH64 = 1025
	R_AARCH64_JUMP_SLOT                       R_AARCH64 = 1026
	R_AARCH64_RELATIVE                        R_AARCH64 = 1027
	R_AARCH64_TLS_DTPMOD64                    R_AARCH64 = 1028
	R_AARCH64_TLS_DTPREL64                    R_AARCH64 = 1029
	R_AARCH64_TLS_TPREL64                     R_AARCH64 = 1030
	R_AARCH64_TLSDESC                         R_AARCH64 = 1031
	R_AARCH64_IRELATIVE                       R_AARCH64 = 1032
)

var raarch64Strings = []intName{
	{0, "R_AARCH64_NONE"},
	{1, "R_AARCH64_P32_ABS32"},
	{2, "R_AARCH64_P32_ABS16"},
	{3, "R_AARCH64_P32_PREL32"},
	{4, "R_AARCH64_P32_PREL16"},
	{5, "R_AARCH64_P32_MOVW_UABS_G0"},
	{6, "R_AARCH64_P32_MOVW_UABS_G0_NC"},
	{7, "R_AARCH64_P32_MOVW_UABS_G1"},
	{8, "R_AARCH64_P32_MOVW_SABS_G0"},
	{9, "R_AARCH64_P32_LD_PREL_LO19"},
	{10, "R_AARCH64_P32_ADR_PREL_LO21"},
	{11, "R_AARCH64_P32_ADR_PREL_PG_HI21"},
	{12, "R_AARCH64_P32_ADD_ABS_LO12_NC"},
	{13, "R_AARCH64_P32_LDST8_ABS_LO12_NC"},
	{14, "R_AARCH64_P32_LDST16_ABS_LO12_NC"},
	{15, "R_AARCH64_P32_LDST32_ABS_LO12_NC"},
	{16, "R_AARCH64_P32_LDST64_ABS_LO12_NC"},
	{17, "R_AARCH64_P32_LDST128_ABS_LO12_NC"},
	{18, "R_AARCH64_P32_TSTBR14"},
	{19, "R_AARCH64_P32_CONDBR19"},
	{20, "R_AARCH64_P32_JUMP26"},
	{21, "R_AARCH64_P32_CALL26"},
	{25, "R_AARCH64_P32_GOT_LD_PREL19"},
	{26, "R_AARCH64_P32_ADR_GOT_PAGE"},
	{27, "R_AARCH64_P32_LD32_GOT_LO12_NC"},
	{81, "R_AARCH64_P32_TLSGD_ADR_PAGE21"},
	{82, "R_AARCH64_P32_TLSGD_ADD_LO12_NC"},
	{103, "R_AARCH64_P32_TLSIE_ADR_GOTTPREL_PAGE21"},
	{104, "R_AARCH64_P32_TLSIE_LD32_GOTTPREL_LO12_NC"},
	{105, "R_AARCH64_P32_TLSIE_LD_GOTTPREL_PREL19"},
	{106, "R_AARCH64_P32_TLSLE_MOVW_TPREL_G1"},
	{107, "R_AARCH64_P32_TLSLE_MOVW_TPREL_G0"},
	{108, "R_AARCH64_P32_TLSLE_MOVW_TPREL_G0_NC"},
	{109, "R_AARCH64_P32_TLSLE_ADD_TPREL_HI12"},
	{110, "R_AARCH64_P32_TLSLE_ADD_TPREL_LO12"},
	{111, "R_AARCH64_P32_TLSLE_ADD_TPREL_LO12_NC"},
	{122, "R_AARCH64_P32_TLSDESC_LD_PREL19"},
	{123, "R_AARCH64_P32_TLSDESC_ADR_PREL21"},
	{124, "R_AARCH64_P32_TLSDESC_ADR_PAGE21"},
	{125, "R_AARCH64_P32_TLSDESC_LD32_LO12_NC"},
	{126, "R_AARCH64_P32_TLSDESC_ADD_LO12_NC"},
	{127, "R_AARCH64_P32_TLSDESC_CALL"},
	{180, "R_AARCH64_P32_COPY"},
	{181, "R_AARCH64_P32_GLOB_DAT"},
	{182, "R_AARCH64_P32_JUMP_SLOT"},
	{183, "R_AARCH64_P32_RELATIVE"},
	{184, "R_AARCH64_P32_TLS_DTPMOD"},
	{185, "R_AARCH64_P32_TLS_DTPREL"},
	{186, "R_AARCH64_P32_TLS_TPREL"},
	{187, "R_AARCH64_P32_TLSDESC"},
	{188, "R_AARCH64_P32_IRELATIVE"},
	{256, "R_AARCH64_NULL"},
	{257, "R_AARCH64_ABS64"},
	{258, "R_AARCH64_ABS32"},
	{259, "R_AARCH64_ABS16"},
	{260, "R_AARCH64_PREL64"},
	{261, "R_AARCH64_PREL32"},
	{262, "R_AARCH64_PREL16"},
	{263, "R_AARCH64_MOVW_UABS_G0"},
	{264, "R_AARCH64_MOVW_UABS_G0_NC"},
	{265, "R_AARCH64_MOVW_UABS_G1"},
	{266, "R_AARCH64_MOVW_UABS_G1_NC"},
	{267, "R_AARCH64_MOVW_UABS_G2"},
	{268, "R_AARCH64_MOVW_UABS_G2_NC"},
	{269, "R_AARCH64_MOVW_UABS_G3"},
	{270, "R_AARCH64_MOVW_SABS_G0"},
	{271, "R_AARCH64_MOVW_SABS_G1"},
	{272, "R_AARCH64_MOVW_SABS_G2"},
	{273, "R_AARCH64_LD_PREL_LO19"},
	{274, "R_AARCH64_ADR_PREL_LO21"},
	{275, "R_AARCH64_ADR_PREL_PG_HI21"},
	{276, "R_AARCH64_ADR_PREL_PG_HI21_NC"},
	{277, "R_AARCH64_ADD_ABS_LO12_NC"},
	{278, "R_AARCH64_LDST8_ABS_LO12_NC"},
	{279, "R_AARCH64_TSTBR14"},
	{280, "R_AARCH64_CONDBR19"},
	{282, "R_AARCH64_JUMP26"},
	{283, "R_AARCH64_CALL26"},
	{284, "R_AARCH64_LDST16_ABS_LO12_NC"},
	{285, "R_AARCH64_LDST32_ABS_LO12_NC"},
	{286, "R_AARCH64_LDST64_ABS_LO12_NC"},
	{299, "R_AARCH64_LDST128_ABS_LO12_NC"},
	{309, "R_AARCH64_GOT_LD_PREL19"},
	{310, "R_AARCH64_LD64_GOTOFF_LO15"},
	{311, "R_AARCH64_ADR_GOT_PAGE"},
	{312, "R_AARCH64_LD64_GOT_LO12_NC"},
	{313, "R_AARCH64_LD64_GOTPAGE_LO15"},
	{512, "R_AARCH64_TLSGD_ADR_PREL21"},
	{513, "R_AARCH64_TLSGD_ADR_PAGE21"},
	{514, "R_AARCH64_TLSGD_ADD_LO12_NC"},
	{515, "R_AARCH64_TLSGD_MOVW_G1"},
	{516, "R_AARCH64_TLSGD_MOVW_G0_NC"},
	{517, "R_AARCH64_TLSLD_ADR_PREL21"},
	{518, "R_AARCH64_TLSLD_ADR_PAGE21"},
	{539, "R_AARCH64_TLSIE_MOVW_GOTTPREL_G1"},
	{540, "R_AARCH64_TLSIE_MOVW_GOTTPREL_G0_NC"},
	{541, "R_AARCH64_TLSIE_ADR_GOTTPREL_PAGE21"},
	{542, "R_AARCH64_TLSIE_LD64_GOTTPREL_LO12_NC"},
	{543, "R_AARCH64_TLSIE_LD_GOTTPREL_PREL19"},
	{544, "R_AARCH64_TLSLE_MOVW_TPREL_G2"},
	{545, "R_AARCH64_TLSLE_MOVW_TPREL_G1"},
	{546, "R_AARCH64_TLSLE_MOVW_TPREL_G1_NC"},
	{547, "R_AARCH64_TLSLE_MOVW_TPREL_G0"},
	{548, "R_AARCH64_TLSLE_MOVW_TPREL_G0_NC"},
	{549, "R_AARCH64_TLSLE_ADD_TPREL_HI12"},
	{550, "R_AARCH64_TLSLE_ADD_TPREL_LO12"},
	{551, "R_AARCH64_TLSLE_ADD_TPREL_LO12_NC"},
	{560, "R_AARCH64_TLSDESC_LD_PREL19"},
	{561, "R_AARCH64_TLSDESC_ADR_PREL21"},
	{562, "R_AARCH64_TLSDESC_ADR_PAGE21"},
	{563, "R_AARCH64_TLSDESC_LD64_LO12_NC"},
	{564, "R_AARCH64_TLSDESC_ADD_LO12_NC"},
	{565, "R_AARCH64_TLSDESC_OFF_G1"},
	{566, "R_AARCH64_TLSDESC_OFF_G0_NC"},
	{567, "R_AARCH64_TLSDESC_LDR"},
	{568, "R_AARCH64_TLSDESC_ADD"},
	{569, "R_AARCH64_TLSDESC_CALL"},
	{570, "R_AARCH64_TLSLE_LDST128_TPREL_LO12"},
	{571, "R_AARCH64_TLSLE_LDST128_TPREL_LO12_NC"},
	{572, "R_AARCH64_TLSLD_LDST128_DTPREL_LO12"},
	{573, "R_AARCH64_TLSLD_LDST128_DTPREL_LO12_NC"},
	{1024, "R_AARCH64_COPY"},
	{1025, "R_AARCH64_GLOB_DAT"},
	{1026, "R_AARCH64_JUMP_SLOT"},
	{1027, "R_AARCH64_RELATIVE"},
	{1028, "R_AARCH64_TLS_DTPMOD64"},
	{1029, "R_AARCH64_TLS_DTPREL64"},
	{1030, "R_AARCH64_TLS_TPREL64"},
	{1031, "R_AARCH64_TLSDESC"},
	{1032, "R_AARCH64_IRELATIVE"},
}

func (i R_AARCH64) String() string   { return stringName(uint32(i), raarch64Strings, false) }
func (i R_AARCH64) GoString() string { return stringName(uint32(i), raarch64Strings, true) }

// Relocation types for Alpha.
type R_ALPHA int

const (
	R_ALPHA_NONE           R_ALPHA = 0  /* No reloc */
	R_ALPHA_REFLONG        R_ALPHA = 1  /* Direct 32 bit */
	R_ALPHA_REFQUAD        R_ALPHA = 2  /* Direct 64 bit */
	R_ALPHA_GPREL32        R_ALPHA = 3  /* GP relative 32 bit */
	R_ALPHA_LITERAL        R_ALPHA = 4  /* GP relative 16 bit w/optimization */
	R_ALPHA_LITUSE         R_ALPHA = 5  /* Optimization hint for LITERAL */
	R_ALPHA_GPDISP         R_ALPHA = 6  /* Add displacement to GP */
	R_ALPHA_BRADDR         R_ALPHA = 7  /* PC+4 relative 23 bit shifted */
	R_ALPHA_HINT           R_ALPHA = 8  /* PC+4 relative 16 bit shifted */
	R_ALPHA_SREL16         R_ALPHA = 9  /* PC relative 16 bit */
	R_ALPHA_SREL32         R_ALPHA = 10 /* PC relative 32 bit */
	R_ALPHA_SREL64         R_ALPHA = 11 /* PC relative 64 bit */
	R_ALPHA_OP_PUSH        R_ALPHA = 12 /* OP stack push */
	R_ALPHA_OP_STORE       R_ALPHA = 13 /* OP stack pop and store */
	R_ALPHA_OP_PSUB        R_ALPHA = 14 /* OP stack subtract */
	R_ALPHA_OP_PRSHIFT     R_ALPHA = 15 /* OP stack right shift */
	R_ALPHA_GPVALUE        R_ALPHA = 16
	R_ALPHA_GPRELHIGH      R_ALPHA = 17
	R_ALPHA_GPRELLOW       R_ALPHA = 18
	R_ALPHA_IMMED_GP_16    R_ALPHA = 19
	R_ALPHA_IMMED_GP_HI32  R_ALPHA = 20
	R_ALPHA_IMMED_SCN_HI32 R_ALPHA = 21
	R_ALPHA_IMMED_BR_HI32  R_ALPHA = 22
	R_ALPHA_IMMED_LO32     R_ALPHA = 23
	R_ALPHA_COPY           R_ALPHA = 24 /* Copy symbol at runtime */
	R_ALPHA_GLOB_DAT       R_ALPHA = 25 /* Create GOT entry */
	R_ALPHA_JMP_SLOT       R_ALPHA = 26 /* Create PLT entry */
	R_ALPHA_RELATIVE       R_ALPHA = 27 /* Adjust by program base */
)

var ralphaStrings = []intName{
	{0, "R_ALPHA_NONE"},
	{1, "R_ALPHA_REFLONG"},
	{2, "R_ALPHA_REFQUAD"},
	{3, "R_ALPHA_GPREL32"},
	{4, "R_ALPHA_LITERAL"},
	{5, "R_ALPHA_LITUSE"},
	{6, "R_ALPHA_GPDISP"},
	{7, "R_ALPHA_BRADDR"},
	{8, "R_ALPHA_HINT"},
	{9, "R_ALPHA_SREL16"},
	{10, "R_ALPHA_SREL32"},
	{11, "R_ALPHA_SREL64"},
	{12, "R_ALPHA_OP_PUSH"},
	{13, "R_ALPHA_OP_STORE"},
	{14, "R_ALPHA_OP_PSUB"},
	{15, "R_ALPHA_OP_PRSHIFT"},
	{16, "R_ALPHA_GPVALUE"},
	{17, "R_ALPHA_GPRELHIGH"},
	{18, "R_ALPHA_GPRELLOW"},
	{19, "R_ALPHA_IMMED_GP_16"},
	{20, "R_ALPHA_IMMED_GP_HI32"},
	{21, "R_ALPHA_IMMED_SCN_HI32"},
	{22, "R_ALPHA_IMMED_BR_HI32"},
	{23, "R_ALPHA_IMMED_LO32"},
	{24, "R_ALPHA_COPY"},
	{25, "R_ALPHA_GLOB_DAT"},
	{26, "R_ALPHA_JMP_SLOT"},
	{27, "R_ALPHA_RELATIVE"},
}

func (i R_ALPHA) String() string   { return stringName(uint32(i), ralphaStrings, false) }
func (i R_ALPHA) GoString() string { return stringName(uint32(i), ralphaStrings, true) }

// Relocation types for ARM.
type R_ARM int

const (
	R_ARM_NONE               R_ARM = 0 /* No relocation. */
	R_ARM_PC24               R_ARM = 1
	R_ARM_ABS32              R_ARM = 2
	R_ARM_REL32              R_ARM = 3
	R_ARM_PC13               R_ARM = 4
	R_ARM_ABS16              R_ARM = 5
	R_ARM_ABS12              R_ARM = 6
	R_ARM_THM_ABS5           R_ARM = 7
	R_ARM_ABS8               R_ARM = 8
	R_ARM_SBREL32            R_ARM = 9
	R_ARM_THM_PC22           R_ARM = 10
	R_ARM_THM_PC8            R_ARM = 11
	R_ARM_AMP_VCALL9         R_ARM = 12
	R_ARM_SWI24              R_ARM = 13
	R_ARM_THM_SWI8           R_ARM = 14
	R_ARM_XPC25              R_ARM = 15
	R_ARM_THM_XPC22          R_ARM = 16
	R_ARM_TLS_DTPMOD32       R_ARM = 17
	R_ARM_TLS_DTPOFF32       R_ARM = 18
	R_ARM_TLS_TPOFF32        R_ARM = 19
	R_ARM_COPY               R_ARM = 20 /* Copy data from shared object. */
	R_ARM_GLOB_DAT           R_ARM = 21 /* Set GOT entry to data address. */
	R_ARM_JUMP_SLOT          R_ARM = 22 /* Set GOT entry to code address. */
	R_ARM_RELATIVE           R_ARM = 23 /* Add load address of shared object. */
	R_ARM_GOTOFF             R_ARM = 24 /* Add GOT-relative symbol address. */
	R_ARM_GOTPC              R_ARM = 25 /* Add PC-relative GOT table address. */
	R_ARM_GOT32              R_ARM = 26 /* Add PC-relative GOT offset. */
	R_ARM_PLT32              R_ARM = 27 /* Add PC-relative PLT offset. */
	R_ARM_CALL               R_ARM = 28
	R_ARM_JUMP24             R_ARM = 29
	R_ARM_THM_JUMP24         R_ARM = 30
	R_ARM_BASE_ABS           R_ARM = 31
	R_ARM_ALU_PCREL_7_0      R_ARM = 32
	R_ARM_ALU_PCREL_15_8     R_ARM = 33
	R_ARM_ALU_PCREL_23_15    R_ARM = 34
	R_ARM_LDR_SBREL_11_10_NC R_ARM = 35
	R_ARM_ALU_SBREL_19_12_NC R_ARM = 36
	R_ARM_ALU_SBREL_27_20_CK R_ARM = 37
	R_ARM_TARGET1            R_ARM = 38
	R_ARM_SBREL31            R_ARM = 39
	R_ARM_V4BX               R_ARM = 40
	R_ARM_TARGET2            R_ARM = 41
	R_ARM_PREL31             R_ARM = 42
	R_ARM_MOVW_ABS_NC        R_ARM = 43
	R_ARM_MOVT_ABS           R_ARM = 44
	R_ARM_MOVW_PREL_NC       R_ARM = 45
	R_ARM_MOVT_PREL          R_ARM = 46
	R_ARM_THM_MOVW_ABS_NC    R_ARM = 47
	R_ARM_THM_MOVT_ABS       R_ARM = 48
	R_ARM_THM_MOVW_PREL_NC   R_ARM = 49
	R_ARM_THM_MOVT_PREL      R_ARM = 50
	R_ARM_THM_JUMP19         R_ARM = 51
	R_ARM_THM_JUMP6          R_ARM = 52
	R_ARM_THM_ALU_PREL_11_0  R_ARM = 53
	R_ARM_THM_PC12           R_ARM = 54
	R_ARM_ABS32_NOI          R_ARM = 55
	R_ARM_REL32_NOI          R_ARM = 56
	R_ARM_ALU_PC_G0_NC       R_ARM = 57
	R_ARM_ALU_PC_G0          R_ARM = 58
	R_ARM_ALU_PC_G1_NC       R_ARM = 59
	R_ARM_ALU_PC_G1          R_ARM = 60
	R_ARM_ALU_PC_G2          R_ARM = 61
	R_ARM_LDR_PC_G1          R_ARM = 62
	R_ARM_LDR_PC_G2          R_ARM = 63
	R_ARM_LDRS_PC_G0         R_ARM = 64
	R_ARM_LDRS_PC_G1         R_ARM = 65
	R_ARM_LDRS_PC_G2         R_ARM = 66
	R_ARM_LDC_PC_G0          R_ARM = 67
	R_ARM_LDC_PC_G1          R_ARM = 68
	R_ARM_LDC_PC_G2          R_ARM = 69
	R_ARM_ALU_SB_G0_NC       R_ARM = 70
	R_ARM_ALU_SB_G0          R_ARM = 71
	R_ARM_ALU_SB_G1_NC       R_ARM = 72
	R_ARM_ALU_SB_G1          R_ARM = 73
	R_ARM_ALU_SB_G2          R_ARM = 74
	R_ARM_LDR_SB_G0          R_ARM = 75
	R_ARM_LDR_SB_G1          R_ARM = 76
	R_ARM_LDR_SB_G2          R_ARM = 77
	R_ARM_LDRS_SB_G0         R_ARM = 78
	R_ARM_LDRS_SB_G1         R_ARM = 79
	R_ARM_LDRS_SB_G2         R_ARM = 80
	R_ARM_LDC_SB_G0          R_ARM = 81
	R_ARM_LDC_SB_G1          R_ARM = 82
	R_ARM_LDC_SB_G2          R_ARM = 83
	R_ARM_MOVW_BREL_NC       R_ARM = 84
	R_ARM_MOVT_BREL          R_ARM = 85
	R_ARM_MOVW_BREL          R_ARM = 86
	R_ARM_THM_MOVW_BREL_NC   R_ARM = 87
	R_ARM_THM_MOVT_BREL      R_ARM = 88
	R_ARM_THM_MOVW_BREL      R_ARM = 89
	R_ARM_TLS_GOTDESC        R_ARM = 90
	R_ARM_TLS_CALL           R_ARM = 91
	R_ARM_TLS_DESCSEQ        R_ARM = 92
	R_ARM_THM_TLS_CALL       R_ARM = 93
	R_ARM_PLT32_ABS          R_ARM = 94
	R_ARM_GOT_ABS            R_ARM = 95
	R_ARM_GOT_PREL           R_ARM = 96
	R_ARM_GOT_BREL12         R_ARM = 97
	R_ARM_GOTOFF12           R_ARM = 98
	R_ARM_GOTRELAX           R_ARM = 99
	R_ARM_GNU_VTENTRY        R_ARM = 100
	R_ARM_GNU_VTINHERIT      R_ARM = 101
	R_ARM_THM_JUMP11         R_ARM = 102
	R_ARM_THM_JUMP8          R_ARM = 103
	R_ARM_TLS_GD32           R_ARM = 104
	R_ARM_TLS_LDM32          R_ARM = 105
	R_ARM_TLS_LDO32          R_ARM = 106
	R_ARM_TLS_IE32           R_ARM = 107
	R_ARM_TLS_LE32           R_ARM = 108
	R_ARM_TLS_LDO12          R_ARM = 109
	R_ARM_TLS_LE12           R_ARM = 110
	R_ARM_TLS_IE12GP         R_ARM = 111
	R_ARM_PRIVATE_0          R_ARM = 112
	R_ARM_PRIVATE_1          R_ARM = 113
	R_ARM_PRIVATE_2          R_ARM = 114
	R_ARM_PRIVATE_3          R_ARM = 115
	R_ARM_PRIVATE_4          R_ARM = 116
	R_ARM_PRIVATE_5          R_ARM = 117
	R_ARM_PRIVATE_6          R_ARM = 118
	R_ARM_PRIVATE_7          R_ARM = 119
	R_ARM_PRIVATE_8          R_ARM = 120
	R_ARM_PRIVATE_9          R_ARM = 121
	R_ARM_PRIVATE_10         R_ARM = 122
	R_ARM_PRIVATE_11         R_ARM = 123
	R_ARM_PRIVATE_12         R_ARM = 124
	R_ARM_PRIVATE_13         R_ARM = 125
	R_ARM_PRIVATE_14         R_ARM = 126
	R_ARM_PRIVATE_15         R_ARM = 127
	R_ARM_ME_TOO             R_ARM = 128
	R_ARM_THM_TLS_DESCSEQ16  R_ARM = 129
	R_ARM_THM_TLS_DESCSEQ32  R_ARM = 130
	R_ARM_THM_GOT_BREL12     R_ARM = 131
	R_ARM_THM_ALU_ABS_G0_NC  R_ARM = 132
	R_ARM_THM_ALU_ABS_G1_NC  R_ARM = 133
	R_ARM_THM_ALU_ABS_G2_NC  R_ARM = 134
	R_ARM_THM_ALU_ABS_G3     R_ARM = 135
	R_ARM_IRELATIVE          R_ARM = 160
	R_ARM_RXPC25             R_ARM = 249
	R_ARM_RSBREL32           R_ARM = 250
	R_ARM_THM_RPC22          R_ARM = 251
	R_ARM_RREL32             R_ARM = 252
	R_ARM_RABS32             R_ARM = 253
	R_ARM_RPC24              R_ARM = 254
	R_ARM_RBASE              R_ARM = 255
)

var rarmStrings = []intName{
	{0, "R_ARM_NONE"},
	{1, "R_ARM_PC24"},
	{2, "R_ARM_ABS32"},
	{3, "R_ARM_REL32"},
	{4, "R_ARM_PC13"},
	{5, "R_ARM_ABS16"},
	{6, "R_ARM_ABS12"},
	{7, "R_ARM_THM_ABS5"},
	{8, "R_ARM_ABS8"},
	{9, "R_ARM_SBREL32"},
	{10, "R_ARM_THM_PC22"},
	{11, "R_ARM_THM_PC8"},
	{12, "R_ARM_AMP_VCALL9"},
	{13, "R_ARM_SWI24"},
	{14, "R_ARM_THM_SWI8"},
	{15, "R_ARM_XPC25"},
	{16, "R_ARM_THM_XPC22"},
	{17, "R_ARM_TLS_DTPMOD32"},
	{18, "R_ARM_TLS_DTPOFF32"},
	{19, "R_ARM_TLS_TPOFF32"},
	{20, "R_ARM_COPY"},
	{21, "R_ARM_GLOB_DAT"},
	{22, "R_ARM_JUMP_SLOT"},
	{23, "R_ARM_RELATIVE"},
	{24, "R_ARM_GOTOFF"},
	{25, "R_ARM_GOTPC"},
	{26, "R_ARM_GOT32"},
	{27, "R_ARM_PLT32"},
	{28, "R_ARM_CALL"},
	{29, "R_ARM_JUMP24"},
	{30, "R_ARM_THM_JUMP24"},
	{31, "R_ARM_BASE_ABS"},
	{32, "R_ARM_ALU_PCREL_7_0"},
	{33, "R_ARM_ALU_PCREL_15_8"},
	{34, "R_ARM_ALU_PCREL_23_15"},
	{35, "R_ARM_LDR_SBREL_11_10_NC"},
	{36, "R_ARM_ALU_SBREL_19_12_NC"},
	{37, "R_ARM_ALU_SBREL_27_20_CK"},
	{38, "R_ARM_TARGET1"},
	{39, "R_ARM_SBREL31"},
	{40, "R_ARM_V4BX"},
	{41, "R_ARM_TARGET2"},
	{42, "R_ARM_PREL31"},
	{43, "R_ARM_MOVW_ABS_NC"},
	{44, "R_ARM_MOVT_ABS"},
	{45, "R_ARM_MOVW_PREL_NC"},
	{46, "R_ARM_MOVT_PREL"},
	{47, "R_ARM_THM_MOVW_ABS_NC"},
	{48, "R_ARM_THM_MOVT_ABS"},
	{49, "R_ARM_THM_MOVW_PREL_NC"},
	{50, "R_ARM_THM_MOVT_PREL"},
	{51, "R_ARM_THM_JUMP19"},
	{52, "R_ARM_THM_JUMP6"},
	{53, "R_ARM_THM_ALU_PREL_11_0"},
	{54, "R_ARM_THM_PC12"},
	{55, "R_ARM_ABS32_NOI"},
	{56, "R_ARM_REL32_NOI"},
	{57, "R_ARM_ALU_PC_G0_NC"},
	{58, "R_ARM_ALU_PC_G0"},
	{59, "R_ARM_ALU_PC_G1_NC"},
	{60, "R_ARM_ALU_PC_G1"},
	{61, "R_ARM_ALU_PC_G2"},
	{62, "R_ARM_LDR_PC_G1"},
	{63, "R_ARM_LDR_PC_G2"},
	{64, "R_ARM_LDRS_PC_G0"},
	{65, "R_ARM_LDRS_PC_G1"},
	{66, "R_ARM_LDRS_PC_G2"},
	{67, "R_ARM_LDC_PC_G0"},
	{68, "R_ARM_LDC_PC_G1"},
	{69, "R_ARM_LDC_PC_G2"},
	{70, "R_ARM_ALU_SB_G0_NC"},
	{71, "R_ARM_ALU_SB_G0"},
	{72, "R_ARM_ALU_SB_G1_NC"},
	{73, "R_ARM_ALU_SB_G1"},
	{74, "R_ARM_ALU_SB_G2"},
	{75, "R_ARM_LDR_SB_G0"},
	{76, "R_ARM_LDR_SB_G1"},
	{77, "R_ARM_LDR_SB_G2"},
	{78, "R_ARM_LDRS_SB_G0"},
	{79, "R_ARM_LDRS_SB_G1"},
	{80, "R_ARM_LDRS_SB_G2"},
	{81, "R_ARM_LDC_SB_G0"},
	{82, "R_ARM_LDC_SB_G1"},
	{83, "R_ARM_LDC_SB_G2"},
	{84, "R_ARM_MOVW_BREL_NC"},
	{85, "R_ARM_MOVT_BREL"},
	{86, "R_ARM_MOVW_BREL"},
	{87, "R_ARM_THM_MOVW_BREL_NC"},
	{88, "R_ARM_THM_MOVT_BREL"},
	{89, "R_ARM_THM_MOVW_BREL"},
	{90, "R_ARM_TLS_GOTDESC"},
	{91, "R_ARM_TLS_CALL"},
	{92, "R_ARM_TLS_DESCSEQ"},
	{93, "R_ARM_THM_TLS_CALL"},
	{94, "R_ARM_PLT32_ABS"},
	{95, "R_ARM_GOT_ABS"},
	{96, "R_ARM_GOT_PREL"},
	{97, "R_ARM_GOT_BREL12"},
	{98, "R_ARM_GOTOFF12"},
	{99, "R_ARM_GOTRELAX"},
	{100, "R_ARM_GNU_VTENTRY"},
	{101, "R_ARM_GNU_VTINHERIT"},
	{102, "R_ARM_THM_JUMP11"},
	{103, "R_ARM_THM_JUMP8"},
	{104, "R_ARM_TLS_GD32"},
	{105, "R_ARM_TLS_LDM32"},
	{106, "R_ARM_TLS_LDO32"},
	{107, "R_ARM_TLS_IE32"},
	{108, "R_ARM_TLS_LE32"},
	{109, "R_ARM_TLS_LDO12"},
	{110, "R_ARM_TLS_LE12"},
	{111, "R_ARM_TLS_IE12GP"},
	{112, "R_ARM_PRIVATE_0"},
	{113, "R_ARM_PRIVATE_1"},
	{114, "R_ARM_PRIVATE_2"},
	{115, "R_ARM_PRIVATE_3"},
	{116, "R_ARM_PRIVATE_4"},
	{117, "R_ARM_PRIVATE_5"},
	{118, "R_ARM_PRIVATE_6"},
	{119, "R_ARM_PRIVATE_7"},
	{120, "R_ARM_PRIVATE_8"},
	{121, "R_ARM_PRIVATE_9"},
	{122, "R_ARM_PRIVATE_10"},
	{123, "R_ARM_PRIVATE_11"},
	{124, "R_ARM_PRIVATE_12"},
	{125, "R_ARM_PRIVATE_13"},
	{126, "R_ARM_PRIVATE_14"},
	{127, "R_ARM_PRIVATE_15"},
	{128, "R_ARM_ME_TOO"},
	{129, "R_ARM_THM_TLS_DESCSEQ16"},
	{130, "R_ARM_THM_TLS_DESCSEQ32"},
	{131, "R_ARM_THM_GOT_BREL12"},
	{132, "R_ARM_THM_ALU_ABS_G0_NC"},
	{133, "R_ARM_THM_ALU_ABS_G1_NC"},
	{134, "R_ARM_THM_ALU_ABS_G2_NC"},
	{135, "R_ARM_THM_ALU_ABS_G3"},
	{160, "R_ARM_IRELATIVE"},
	{249, "R_ARM_RXPC25"},
	{250, "R_ARM_RSBREL32"},
	{251, "R_ARM_THM_RPC22"},
	{252, "R_ARM_RREL32"},
	{253, "R_ARM_RABS32"},
	{254, "R_ARM_RPC24"},
	{255, "R_ARM_RBASE"},
}

func (i R_ARM) String() string   { return stringName(uint32(i), rarmStrings, false) }
func (i R_ARM) GoString() string { return stringName(uint32(i), rarmStrings, true) }

// Relocation types for 386.
type R_386 int

const (
	R_386_NONE          R_386 = 0  /* No relocation. */
	R_386_32            R_386 = 1  /* Add symbol value. */
	R_386_PC32          R_386 = 2  /* Add PC-relative symbol value. */
	R_386_GOT32         R_386 = 3  /* Add PC-relative GOT offset. */
	R_386_PLT32         R_386 = 4  /* Add PC-relative PLT offset. */
	R_386_COPY          R_386 = 5  /* Copy data from shared object. */
	R_386_GLOB_DAT      R_386 = 6  /* Set GOT entry to data address. */
	R_386_JMP_SLOT      R_386 = 7  /* Set GOT entry to code address. */
	R_386_RELATIVE      R_386 = 8  /* Add load address of shared object. */
	R_386_GOTOFF        R_386 = 9  /* Add GOT-relative symbol address. */
	R_386_GOTPC         R_386 = 10 /* Add PC-relative GOT table address. */
	R_386_32PLT         R_386 = 11
	R_386_TLS_TPOFF     R_386 = 14 /* Negative offset in static TLS block */
	R_386_TLS_IE        R_386 = 15 /* Absolute address of GOT for -ve static TLS */
	R_386_TLS_GOTIE     R_386 = 16 /* GOT entry for negative static TLS block */
	R_386_TLS_LE        R_386 = 17 /* Negative offset relative to static TLS */
	R_386_TLS_GD        R_386 = 18 /* 32 bit offset to GOT (index,off) pair */
	R_386_TLS_LDM       R_386 = 19 /* 32 bit offset to GOT (index,zero) pair */
	R_386_16            R_386 = 20
	R_386_PC16          R_386 = 21
	R_386_8             R_386 = 22
	R_386_PC8           R_386 = 23
	R_386_TLS_GD_32     R_386 = 24 /* 32 bit offset to GOT (index,off) pair */
	R_386_TLS_GD_PUSH   R_386 = 25 /* pushl instruction for Sun ABI GD sequence */
	R_386_TLS_GD_CALL   R_386 = 26 /* call instruction for Sun ABI GD sequence */
	R_386_TLS_GD_POP    R_386 = 27 /* popl instruction for Sun ABI GD sequence */
	R_386_TLS_LDM_32    R_386 = 28 /* 32 bit offset to GOT (index,zero) pair */
	R_386_TLS_LDM_PUSH  R_386 = 29 /* pushl instruction for Sun ABI LD sequence */
	R_386_TLS_LDM_CALL  R_386 = 30 /* call instruction for Sun ABI LD sequence */
	R_386_TLS_LDM_POP   R_386 = 31 /* popl instruction for Sun ABI LD sequence */
	R_386_TLS_LDO_32    R_386 = 32 /* 32 bit offset from start of TLS block */
	R_386_TLS_IE_32     R_386 = 33 /* 32 bit offset to GOT static TLS offset entry */
	R_386_TLS_LE_32     R_386 = 34 /* 32 bit offset within static TLS block */
	R_386_TLS_DTPMOD32  R_386 = 35 /* GOT entry containing TLS index */
	R_386_TLS_DTPOFF32  R_386 = 36 /* GOT entry containing TLS offset */
	R_386_TLS_TPOFF32   R_386 = 37 /* GOT entry of -ve static TLS offset */
	R_386_SIZE32        R_386 = 38
	R_386_TLS_GOTDESC   R_386 = 39
	R_386_TLS_DESC_CALL R_386 = 40
	R_386_TLS_DESC      R_386 = 41
	R_386_IRELATIVE     R_386 = 42
	R_386_GOT32X        R_386 = 43
)

var r386Strings = []intName{
	{0, "R_386_NONE"},
	{1, "R_386_32"},
	{2, "R_386_PC32"},
	{3, "R_386_GOT32"},
	{4, "R_386_PLT32"},
	{5, "R_386_COPY"},
	{6, "R_386_GLOB_DAT"},
	{7, "R_386_JMP_SLOT"},
	{8, "R_386_RELATIVE"},
	{9, "R_386_GOTOFF"},
	{10, "R_386_GOTPC"},
	{11, "R_386_32PLT"},
	{14, "R_386_TLS_TPOFF"},
	{15, "R_386_TLS_IE"},
	{16, "R_386_TLS_GOTIE"},
	{17, "R_386_TLS_LE"},
	{18, "R_386_TLS_GD"},
	{19, "R_386_TLS_LDM"},
	{20, "R_386_16"},
	{21, "R_386_PC16"},
	{22, "R_386_8"},
	{23, "R_386_PC8"},
	{24, "R_386_TLS_GD_32"},
	{25, "R_386_TLS_GD_PUSH"},
	{26, "R_386_TLS_GD_CALL"},
	{27, "R_386_TLS_GD_POP"},
	{28, "R_386_TLS_LDM_32"},
	{29, "R_386_TLS_LDM_PUSH"},
	{30, "R_386_TLS_LDM_CALL"},
	{31, "R_386_TLS_LDM_POP"},
	{32, "R_386_TLS_LDO_32"},
	{33, "R_386_TLS_IE_32"},
	{34, "R_386_TLS_LE_32"},
	{35, "R_386_TLS_DTPMOD32"},
	{36, "R_386_TLS_DTPOFF32"},
	{37, "R_386_TLS_TPOFF32"},
	{38, "R_386_SIZE32"},
	{39, "R_386_TLS_GOTDESC"},
	{40, "R_386_TLS_DESC_CALL"},
	{41, "R_386_TLS_DESC"},
	{42, "R_386_IRELATIVE"},
	{43, "R_386_GOT32X"},
}

func (i R_386) String() string   { return stringName(uint32(i), r386Strings, false) }
func (i R_386) GoString() string { return stringName(uint32(i), r386Strings, true) }

// Relocation types for MIPS.
type R_MIPS int

const (
	R_MIPS_NONE          R_MIPS = 0
	R_MIPS_16            R_MIPS = 1
	R_MIPS_32            R_MIPS = 2
	R_MIPS_REL32         R_MIPS = 3
	R_MIPS_26            R_MIPS = 4
	R_MIPS_HI16          R_MIPS = 5  /* high 16 bits of symbol value */
	R_MIPS_LO16          R_MIPS = 6  /* low 16 bits of symbol value */
	R_MIPS_GPREL16       R_MIPS = 7  /* GP-relative reference  */
	R_MIPS_LITERAL       R_MIPS = 8  /* Reference to literal section  */
	R_MIPS_GOT16         R_MIPS = 9  /* Reference to global offset table */
	R_MIPS_PC16          R_MIPS = 10 /* 16 bit PC relative reference */
	R_MIPS_CALL16        R_MIPS = 11 /* 16 bit call through glbl offset tbl */
	R_MIPS_GPREL32       R_MIPS = 12
	R_MIPS_SHIFT5        R_MIPS = 16
	R_MIPS_SHIFT6        R_MIPS = 17
	R_MIPS_64            R_MIPS = 18
	R_MIPS_GOT_DISP      R_MIPS = 19
	R_MIPS_GOT_PAGE      R_MIPS = 20
	R_MIPS_GOT_OFST      R_MIPS = 21
	R_MIPS_GOT_HI16      R_MIPS = 22
	R_MIPS_GOT_LO16      R_MIPS = 23
	R_MIPS_SUB           R_MIPS = 24
	R_MIPS_INSERT_A      R_MIPS = 25
	R_MIPS_INSERT_B      R_MIPS = 26
	R_MIPS_DELETE        R_MIPS = 27
	R_MIPS_HIGHER        R_MIPS = 28
	R_MIPS_HIGHEST       R_MIPS = 29
	R_MIPS_CALL_HI16     R_MIPS = 30
	R_MIPS_CALL_LO16     R_MIPS = 31
	R_MIPS_SCN_DISP      R_MIPS = 32
	R_MIPS_REL16         R_MIPS = 33
	R_MIPS_ADD_IMMEDIATE R_MIPS = 34
	R_MIPS_PJUMP         R_MIPS = 35
	R_MIPS_RELGOT        R_MIPS = 36
	R_MIPS_JALR          R_MIPS = 37

	R_MIPS_TLS_DTPMOD32    R_MIPS = 38 /* Module number 32 bit */
	R_MIPS_TLS_DTPREL32    R_MIPS = 39 /* Module-relative offset 32 bit */
	R_MIPS_TLS_DTPMOD64    R_MIPS = 40 /* Module number 64 bit */
	R_MIPS_TLS_DTPREL64    R_MIPS = 41 /* Module-relative offset 64 bit */
	R_MIPS_TLS_GD          R_MIPS = 42 /* 16 bit GOT offset for GD */
	R_MIPS_TLS_LDM         R_MIPS = 43 /* 16 bit GOT offset for LDM */
	R_MIPS_TLS_DTPREL_HI16 R_MIPS = 44 /* Module-relative offset, high 16 bits */
	R_MIPS_TLS_DTPREL_LO16 R_MIPS = 45 /* Module-relative offset, low 16 bits */
	R_MIPS_TLS_GOTTPREL    R_MIPS = 46 /* 16 bit GOT offset for IE */
	R_MIPS_TLS_TPREL32     R_MIPS = 47 /* TP-relative offset, 32 bit */
	R_MIPS_TLS_TPREL64     R_MIPS = 48 /* TP-relative offset, 64 bit */
	R_MIPS_TLS_TPREL_HI16  R_MIPS = 49 /* TP-relative offset, high 16 bits */
	R_MIPS_TLS_TPREL_LO16  R_MIPS = 50 /* TP-relative offset, low 16 bits */

	R_MIPS_PC32 R_MIPS = 248 /* 32 bit PC relative reference */
)

var rmipsStrings = []intName{
	{0, "R_MIPS_NONE"},
	{1, "R_MIPS_16"},
	{2, "R_MIPS_32"},
	{3, "R_MIPS_REL32"},
	{4, "R_MIPS_26"},
	{5, "R_MIPS_HI16"},
	{6, "R_MIPS_LO16"},
	{7, "R_MIPS_GPREL16"},
	{8, "R_MIPS_LITERAL"},
	{9, "R_MIPS_GOT16"},
	{10, "R_MIPS_PC16"},
	{11, "R_MIPS_CALL16"},
	{12, "R_MIPS_GPREL32"},
	{16, "R_MIPS_SHIFT5"},
	{17, "R_MIPS_SHIFT6"},
	{18, "R_MIPS_64"},
	{19, "R_MIPS_GOT_DISP"},
	{20, "R_MIPS_GOT_PAGE"},
	{21, "R_MIPS_GOT_OFST"},
	{22, "R_MIPS_GOT_HI16"},
	{23, "R_MIPS_GOT_LO16"},
	{24, "R_MIPS_SUB"},
	{25, "R_MIPS_INSERT_A"},
	{26, "R_MIPS_INSERT_B"},
	{27, "R_MIPS_DELETE"},
	{28, "R_MIPS_HIGHER"},
	{29, "R_MIPS_HIGHEST"},
	{30, "R_MIPS_CALL_HI16"},
	{31, "R_MIPS_CALL_LO16"},
	{32, "R_MIPS_SCN_DISP"},
	{33, "R_MIPS_REL16"},
	{34, "R_MIPS_ADD_IMMEDIATE"},
	{35, "R_MIPS_PJUMP"},
	{36, "R_MIPS_RELGOT"},
	{37, "R_MIPS_JALR"},
	{38, "R_MIPS_TLS_DTPMOD32"},
	{39, "R_MIPS_TLS_DTPREL32"},
	{40, "R_MIPS_TLS_DTPMOD64"},
	{41, "R_MIPS_TLS_DTPREL64"},
	{42, "R_MIPS_TLS_GD"},
	{43, "R_MIPS_TLS_LDM"},
	{44, "R_MIPS_TLS_DTPREL_HI16"},
	{45, "R_MIPS_TLS_DTPREL_LO16"},
	{46, "R_MIPS_TLS_GOTTPREL"},
	{47, "R_MIPS_TLS_TPREL32"},
	{48, "R_MIPS_TLS_TPREL64"},
	{49, "R_MIPS_TLS_TPREL_HI16"},
	{50, "R_MIPS_TLS_TPREL_LO16"},
	{248, "R_MIPS_PC32"},
}

func (i R_MIPS) String() string   { return stringName(uint32(i), rmipsStrings, false) }
func (i R_MIPS) GoString() string { return stringName(uint32(i), rmipsStrings, true) }

// Relocation types for LoongArch.
type R_LARCH int

const (
	R_LARCH_NONE                       R_LARCH = 0
	R_LARCH_32                         R_LARCH = 1
	R_LARCH_64                         R_LARCH = 2
	R_LARCH_RELATIVE                   R_LARCH = 3
	R_LARCH_COPY                       R_LARCH = 4
	R_LARCH_JUMP_SLOT                  R_LARCH = 5
	R_LARCH_TLS_DTPMOD32               R_LARCH = 6
	R_LARCH_TLS_DTPMOD64               R_LARCH = 7
	R_LARCH_TLS_DTPREL32               R_LARCH = 8
	R_LARCH_TLS_DTPREL64               R_LARCH = 9
	R_LARCH_TLS_TPREL32                R_LARCH = 10
	R_LARCH_TLS_TPREL64                R_LARCH = 11
	R_LARCH_IRELATIVE                  R_LARCH = 12
	R_LARCH_MARK_LA                    R_LARCH = 20
	R_LARCH_MARK_PCREL                 R_LARCH = 21
	R_LARCH_SOP_PUSH_PCREL             R_LARCH = 22
	R_LARCH_SOP_PUSH_ABSOLUTE          R_LARCH = 23
	R_LARCH_SOP_PUSH_DUP               R_LARCH = 24
	R_LARCH_SOP_PUSH_GPREL             R_LARCH = 25
	R_LARCH_SOP_PUSH_TLS_TPREL         R_LARCH = 26
	R_LARCH_SOP_PUSH_TLS_GOT           R_LARCH = 27
	R_LARCH_SOP_PUSH_TLS_GD            R_LARCH = 28
	R_LARCH_SOP_PUSH_PLT_PCREL         R_LARCH = 29
	R_LARCH_SOP_ASSERT                 R_LARCH = 30
	R_LARCH_SOP_NOT                    R_LARCH = 31
	R_LARCH_SOP_SUB                    R_LARCH = 32
	R_LARCH_SOP_SL                     R_LARCH = 33
	R_LARCH_SOP_SR                     R_LARCH = 34
	R_LARCH_SOP_ADD                    R_LARCH = 35
	R_LARCH_SOP_AND                    R_LARCH = 36
	R_LARCH_SOP_IF_ELSE                R_LARCH = 37
	R_LARCH_SOP_POP_32_S_10_5          R_LARCH = 38
	R_LARCH_SOP_POP_32_U_10_12         R_LARCH = 39
	R_LARCH_SOP_POP_32_S_10_12         R_LARCH = 40
	R_LARCH_SOP_POP_32_S_10_16         R_LARCH = 41
	R_LARCH_SOP_POP_32_S_10_16_S2      R_LARCH = 42
	R_LARCH_SOP_POP_32_S_5_20          R_LARCH = 43
	R_LARCH_SOP_POP_32_S_0_5_10_16_S2  R_LARCH = 44
	R_LARCH_SOP_POP_32_S_0_10_10_16_S2 R_LARCH = 45
	R_LARCH_SOP_POP_32_U               R_LARCH = 46
	R_LARCH_ADD8                       R_LARCH = 47
	R_LARCH_ADD16                      R_LARCH = 48
	R_LARCH_ADD24                      R_LARCH = 49
	R_LARCH_ADD32                      R_LARCH = 50
	R_LARCH_ADD64                      R_LARCH = 51
	R_LARCH_SUB8                       R_LARCH = 52
	R_LARCH_SUB16                      R_LARCH = 53
	R_LARCH_SUB24                      R_LARCH = 54
	R_LARCH_SUB32                      R_LARCH = 55
	R_LARCH_SUB64                      R_LARCH = 56
	R_LARCH_GNU_VTINHERIT              R_LARCH = 57
	R_LARCH_GNU_VTENTRY                R_LARCH = 58
	R_LARCH_B16                        R_LARCH = 64
	R_LARCH_B21                        R_LARCH = 65
	R_LARCH_B26                        R_LARCH = 66
	R_LARCH_ABS_HI20                   R_LARCH = 67
	R_LARCH_ABS_LO12                   R_LARCH = 68
	R_LARCH_ABS64_LO20                 R_LARCH = 69
	R_LARCH_ABS64_HI12                 R_LARCH = 70
	R_LARCH_PCALA_HI20                 R_LARCH = 71
	R_LARCH_PCALA_LO12                 R_LARCH = 72
	R_LARCH_PCALA64_LO20               R_LARCH = 73
	R_LARCH_PCALA64_HI12               R_LARCH = 74
	R_LARCH_GOT_PC_HI20                R_LARCH = 75
	R_LARCH_GOT_PC_LO12                R_LARCH = 76
	R_LARCH_GOT64_PC_LO20              R_LARCH = 77
	R_LARCH_GOT64_PC_HI12              R_LARCH = 78
	R_LARCH_GOT_HI20                   R_LARCH = 79
	R_LARCH_GOT_LO12                   R_LARCH = 80
	R_LARCH_GOT64_LO20                 R_LARCH = 81
	R_LARCH_GOT64_HI12                 R_LARCH = 82
	R_LARCH_TLS_LE_HI20                R_LARCH = 83
	R_LARCH_TLS_LE_LO12                R_LARCH = 84
	R_LARCH_TLS_LE64_LO20              R_LARCH = 85
	R_LARCH_TLS_LE64_HI12              R_LARCH = 86
	R_LARCH_TLS_IE_PC_HI20             R_LARCH = 87
	R_LARCH_TLS_IE_PC_LO12             R_LARCH = 88
	R_LARCH_TLS_IE64_PC_LO20           R_LARCH = 89
	R_LARCH_TLS_IE64_PC_HI12           R_LARCH = 90
	R_LARCH_TLS_IE_HI20                R_LARCH = 91
	R_LARCH_TLS_IE_LO12                R_LARCH = 92
	R_LARCH_TLS_IE64_LO20              R_LARCH = 93
	R_LARCH_TLS_IE64_HI12              R_LARCH = 94
	R_LARCH_TLS_LD_PC_HI20             R_LARCH = 95
	R_LARCH_TLS_LD_HI20                R_LARCH = 96
	R_LARCH_TLS_GD_PC_HI20             R_LARCH = 97
	R_LARCH_TLS_GD_HI20                R_LARCH = 98
	R_LARCH_32_PCREL                   R_LARCH = 99
	R_LARCH_RELAX                      R_LARCH = 100
	R_LARCH_DELETE                     R_LARCH = 101
	R_LARCH_ALIGN                      R_LARCH = 102
	R_LARCH_PCREL20_S2                 R_LARCH = 103
	R_LARCH_CFA                        R_LARCH = 104
	R_LARCH_ADD6                       R_LARCH = 105
	R_LARCH_SUB6                       R_LARCH = 106
	R_LARCH_ADD_ULEB128                R_LARCH = 107
	R_LARCH_SUB_ULEB128                R_LARCH = 108
	R_LARCH_64_PCREL                   R_LARCH = 109
)

var rlarchStrings = []intName{
	{0, "R_LARCH_NONE"},
	{1, "R_LARCH_32"},
	{2, "R_LARCH_64"},
	{3, "R_LARCH_RELATIVE"},
	{4, "R_LARCH_COPY"},
	{5, "R_LARCH_JUMP_SLOT"},
	{6, "R_LARCH_TLS_DTPMOD32"},
	{7, "R_LARCH_TLS_DTPMOD64"},
	{8, "R_LARCH_TLS_DTPREL32"},
	{9, "R_LARCH_TLS_DTPREL64"},
	{10, "R_LARCH_TLS_TPREL32"},
	{11, "R_LARCH_TLS_TPREL64"},
	{12, "R_LARCH_IRELATIVE"},
	{20, "R_LARCH_MARK_LA"},
	{21, "R_LARCH_MARK_PCREL"},
	{22, "R_LARCH_SOP_PUSH_PCREL"},
	{23, "R_LARCH_SOP_PUSH_ABSOLUTE"},
	{24, "R_LARCH_SOP_PUSH_DUP"},
	{25, "R_LARCH_SOP_PUSH_GPREL"},
	{26, "R_LARCH_SOP_PUSH_TLS_TPREL"},
	{27, "R_LARCH_SOP_PUSH_TLS_GOT"},
	{28, "R_LARCH_SOP_PUSH_TLS_GD"},
	{29, "R_LARCH_SOP_PUSH_PLT_PCREL"},
	{30, "R_LARCH_SOP_ASSERT"},
	{31, "R_LARCH_SOP_NOT"},
	{32, "R_LARCH_SOP_SUB"},
	{33, "R_LARCH_SOP_SL"},
	{34, "R_LARCH_SOP_SR"},
	{35, "R_LARCH_SOP_ADD"},
	{36, "R_LARCH_SOP_AND"},
	{37, "R_LARCH_SOP_IF_ELSE"},
	{38, "R_LARCH_SOP_POP_32_S_10_5"},
	{39, "R_LARCH_SOP_POP_32_U_10_12"},
	{40, "R_LARCH_SOP_POP_32_S_10_12"},
	{41, "R_LARCH_SOP_POP_32_S_10_16"},
	{42, "R_LARCH_SOP_POP_32_S_10_16_S2"},
	{43, "R_LARCH_SOP_POP_32_S_5_20"},
	{44, "R_LARCH_SOP_POP_32_S_0_5_10_16_S2"},
	{45, "R_LARCH_SOP_POP_32_S_0_10_10_16_S2"},
	{46, "R_LARCH_SOP_POP_32_U"},
	{47, "R_LARCH_ADD8"},
	{48, "R_LARCH_ADD16"},
	{49, "R_LARCH_ADD24"},
	{50, "R_LARCH_ADD32"},
	{51, "R_LARCH_ADD64"},
	{52, "R_LARCH_SUB8"},
	{53, "R_LARCH_SUB16"},
	{54, "R_LARCH_SUB24"},
	{55, "R_LARCH_SUB32"},
	{56, "R_LARCH_SUB64"},
	{57, "R_LARCH_GNU_VTINHERIT"},
	{58, "R_LARCH_GNU_VTENTRY"},
	{64, "R_LARCH_B16"},
	{65, "R_LARCH_B21"},
	{66, "R_LARCH_B26"},
	{67, "R_LARCH_ABS_HI20"},
	{68, "R_LARCH_ABS_LO12"},
	{69, "R_LARCH_ABS64_LO20"},
	{70, "R_LARCH_ABS64_HI12"},
	{71, "R_LARCH_PCALA_HI20"},
	{72, "R_LARCH_PCALA_LO12"},
	{73, "R_LARCH_PCALA64_LO20"},
	{74, "R_LARCH_PCALA64_HI12"},
	{75, "R_LARCH_GOT_PC_HI20"},
	{76, "R_LARCH_GOT_PC_LO12"},
	{77, "R_LARCH_GOT64_PC_LO20"},
	{78, "R_LARCH_GOT64_PC_HI12"},
	{79, "R_LARCH_GOT_HI20"},
	{80, "R_LARCH_GOT_LO12"},
	{81, "R_LARCH_GOT64_LO20"},
	{82, "R_LARCH_GOT64_HI12"},
	{83, "R_LARCH_TLS_LE_HI20"},
	{84, "R_LARCH_TLS_LE_LO12"},
	{85, "R_LARCH_TLS_LE64_LO20"},
	{86, "R_LARCH_TLS_LE64_HI12"},
	{87, "R_LARCH_TLS_IE_PC_HI20"},
	{88, "R_LARCH_TLS_IE_PC_LO12"},
	{89, "R_LARCH_TLS_IE64_PC_LO20"},
	{90, "R_LARCH_TLS_IE64_PC_HI12"},
	{91, "R_LARCH_TLS_IE_HI20"},
	{92, "R_LARCH_TLS_IE_LO12"},
	{93, "R_LARCH_TLS_IE64_LO20"},
	{94, "R_LARCH_TLS_IE64_HI12"},
	{95, "R_LARCH_TLS_LD_PC_HI20"},
	{96, "R_LARCH_TLS_LD_HI20"},
	{97, "R_LARCH_TLS_GD_PC_HI20"},
	{98, "R_LARCH_TLS_GD_HI20"},
	{99, "R_LARCH_32_PCREL"},
	{100, "R_LARCH_RELAX"},
	{101, "R_LARCH_DELETE"},
	{102, "R_LARCH_ALIGN"},
	{103, "R_LARCH_PCREL20_S2"},
	{104, "R_LARCH_CFA"},
	{105, "R_LARCH_ADD6"},
	{106, "R_LARCH_SUB6"},
	{107, "R_LARCH_ADD_ULEB128"},
	{108, "R_LARCH_SUB_ULEB128"},
	{109, "R_LARCH_64_PCREL"},
}

func (i R_LARCH) String() string   { return stringName(uint32(i), rlarchStrings, false) }
func (i R_LARCH) GoString() string { return stringName(uint32(i), rlarchStrings, true) }

// Relocation types for PowerPC.
//
// Values that are shared by both R_PPC and R_PPC64 are prefixed with
// R_POWERPC_ in the ELF standard. For the R_PPC type, the relevant
// shared relocations have been renamed with the prefix R_PPC_.
// The original name follows the value in a comment.
type R_PPC int

const (
	R_PPC_NONE            R_PPC = 0  // R_POWERPC_NONE
	R_PPC_ADDR32          R_PPC = 1  // R_POWERPC_ADDR32
	R_PPC_ADDR24          R_PPC = 2  // R_POWERPC_ADDR24
	R_PPC_ADDR16          R_PPC = 3  // R_POWERPC_ADDR16
	R_PPC_ADDR16_LO       R_PPC = 4  // R_POWERPC_ADDR16_LO
	R_PPC_ADDR16_HI       R_PPC = 5  // R_POWERPC_ADDR16_HI
	R_PPC_ADDR16_HA       R_PPC = 6  // R_POWERPC_ADDR16_HA
	R_PPC_ADDR14          R_PPC = 7  // R_POWERPC_ADDR14
	R_PPC_ADDR14_BRTAKEN  R_PPC = 8  // R_POWERPC_ADDR14_BRTAKEN
	R_PPC_ADDR14_BRNTAKEN R_PPC = 9  // R_POWERPC_ADDR14_BRNTAKEN
	R_PPC_REL24           R_PPC = 10 // R_POWERPC_REL24
	R_PPC_REL14           R_PPC = 11 // R_POWERPC_REL14
	R_PPC_REL14_BRTAKEN   R_PPC = 12 // R_POWERPC_REL14_BRTAKEN
	R_PPC_REL14_BRNTAKEN  R_PPC = 13 // R_POWERPC_REL14_BRNTAKEN
	R_PPC_GOT16           R_PPC = 14 // R_POWERPC_GOT16
	R_PPC_GOT16_LO        R_PPC = 15 // R_POWERPC_GOT16_LO
	R_PPC_GOT16_HI        R_PPC = 16 // R_POWERPC_GOT16_HI
	R_PPC_GOT16_HA        R_PPC = 17 // R_POWERPC_GOT16_HA
	R_PPC_PLTREL24        R_PPC = 18
	R_PPC_COPY            R_PPC = 19 // R_POWERPC_COPY
	R_PPC_GLOB_DAT        R_PPC = 20 // R_POWERPC_GLOB_DAT
	R_PPC_JMP_SLOT        R_PPC = 21 // R_POWERPC_JMP_SLOT
	R_PPC_RELATIVE        R_PPC = 22 // R_POWERPC_RELATIVE
	R_PPC_LOCAL24PC       R_PPC = 23
	R_PPC_UADDR32         R_PPC = 24 // R_POWERPC_UADDR32
	R_PPC_UADDR16         R_PPC = 25 // R_POWERPC_UADDR16
	R_PPC_REL32           R_PPC = 26 // R_POWERPC_REL32
	R_PPC_PLT32           R_PPC = 27 // R_POWERPC_PLT32
	R_PPC_PLTREL32        R_PPC = 28 // R_POWERPC_PLTREL32
	R_PPC_PLT16_LO        R_PPC = 29 // R_POWERPC_PLT16_LO
	R_PPC_PLT16_HI        R_PPC = 30 // R_POWERPC_PLT16_HI
	R_PPC_PLT16_HA        R_PPC = 31 // R_POWERPC_PLT16_HA
	R_PPC_SDAREL16        R_PPC = 32
	R_PPC_SECTOFF         R_PPC = 33 // R_POWERPC_SECTOFF
	R_PPC_SECTOFF_LO      R_PPC = 34 // R_POWERPC_SECTOFF_LO
	R_PPC_SECTOFF_HI      R_PPC = 35 // R_POWERPC_SECTOFF_HI
	R_PPC_SECTOFF_HA      R_PPC = 36 // R_POWERPC_SECTOFF_HA
	R_PPC_TLS             R_PPC = 67 // R_POWERPC_TLS
	R_PPC_DTPMOD32        R_PPC = 68 // R_POWERPC_DTPMOD32
	R_PPC_TPREL16         R_PPC = 69 // R_POWERPC_TPREL16
	R_PPC_TPREL16_LO      R_PPC = 70 // R_POWERPC_TPREL16_LO
	R_PPC_TPREL16_HI      R_PPC = 71 // R_POWERPC_TPREL16_HI
	R_PPC_TPREL16_HA      R_PPC = 72 // R_POWERPC_TPREL16_HA
	R_PPC_TPREL32         R_PPC = 73 // R_POWERPC_TPREL32
	R_PPC_DTPREL16        R_PPC = 74 // R_POWERPC_DTPREL16
	R_PPC_DTPREL16_LO     R_PPC = 75 // R_POWERPC_DTPREL16_LO
	R_PPC_DTPREL16_HI     R_PPC = 76 // R_POWERPC_DTPREL16_HI
	R_PPC_DTPREL16_HA     R_PPC = 77 // R_POWERPC_DTPREL16_HA
	R_PPC_DTPREL32        R_PPC = 78 // R_POWERPC_DTPREL32
	R_PPC_GOT_TLSGD16     R_PPC = 79 // R_POWERPC_GOT_TLSGD16
	R_PPC_GOT_TLSGD16_LO  R_PPC = 80 // R_POWERPC_GOT_TLSGD16_LO
	R_PPC_GOT_TLSGD16_HI  R_PPC = 81 // R_POWERPC_GOT_TLSGD16_HI
	R_PPC_GOT_TLSGD16_HA  R_PPC = 82 // R_POWERPC_GOT_TLSGD16_HA
	R_PPC_GOT_TLSLD16     R_PPC = 83 // R_POWERPC_GOT_TLSLD16
	R_PPC_GOT_TLSLD16_LO  R_PPC = 84 // R_POWERPC_GOT_TLSLD16_LO
	R_PPC_GOT_TLSLD16_HI  R_PPC = 85 // R_POWERPC_GOT_TLSLD16_HI
	R_PPC_GOT_TLSLD16_HA  R_PPC = 86 // R_POWERPC_GOT_TLSLD16_HA
	R_PPC_GOT_TPREL16     R_PPC = 87 // R_POWERPC_GOT_TPREL16
	R_PPC_GOT_TPREL16_LO  R_PPC = 88 // R_POWERPC_GOT_TPREL16_LO
	R_PPC_GOT_TPREL16_HI  R_PPC = 89 // R_POWERPC_GOT_TPREL16_HI
	R_PPC_GOT_TPREL16_HA  R_PPC = 90 // R_POWERPC_GOT_TPREL16_HA
	R_PPC_EMB_NADDR32     R_PPC = 101
	R_PPC_EMB_NADDR16     R_PPC = 102
	R_PPC_EMB_NADDR16_LO  R_PPC = 103
	R_PPC_EMB_NADDR16_HI  R_PPC = 104
	R_PPC_EMB_NADDR16_HA  R_PPC = 105
	R_PPC_EMB_SDAI16      R_PPC = 106
	R_PPC_EMB_SDA2I16     R_PPC = 107
	R_PPC_EMB_SDA2REL     R_PPC = 108
	R_PPC_EMB_SDA21       R_PPC = 109
	R_PPC_EMB_MRKREF      R_PPC = 110
	R_PPC_EMB_RELSEC16    R_PPC = 111
	R_PPC_EMB_RELST_LO    R_PPC = 112
	R_PPC_EMB_RELST_HI    R_PPC = 113
	R_PPC_EMB_RELST_HA    R_PPC = 114
	R_PPC_EMB_BIT_FLD     R_PPC = 115
	R_PPC_EMB_RELSDA      R_PPC = 116
)

var rppcStrings = []intName{
	{0, "R_PPC_NONE"},
	{1, "R_PPC_ADDR32"},
	{2, "R_PPC_ADDR24"},
	{3, "R_PPC_ADDR16"},
	{4, "R_PPC_ADDR16_LO"},
	{5, "R_PPC_ADDR16_HI"},
	{6, "R_PPC_ADDR16_HA"},
	{7, "R_PPC_ADDR14"},
	{8, "R_PPC_ADDR14_BRTAKEN"},
	{9, "R_PPC_ADDR14_BRNTAKEN"},
	{10, "R_PPC_REL24"},
	{11, "R_PPC_REL14"},
	{12, "R_PPC_REL14_BRTAKEN"},
	{13, "R_PPC_REL14_BRNTAKEN"},
	{14, "R_PPC_GOT16"},
	{15, "R_PPC_GOT16_LO"},
	{16, "R_PPC_GOT16_HI"},
	{17, "R_PPC_GOT16_HA"},
	{18, "R_PPC_PLTREL24"},
	{19, "R_PPC_COPY"},
	{20, "R_PPC_GLOB_DAT"},
	{21, "R_PPC_JMP_SLOT"},
	{22, "R_PPC_RELATIVE"},
	{23, "R_PPC_LOCAL24PC"},
	{24, "R_PPC_UADDR32"},
	{25, "R_PPC_UADDR16"},
	{26, "R_PPC_REL32"},
	{27, "R_PPC_PLT32"},
	{28, "R_PPC_PLTREL32"},
	{29, "R_PPC_PLT16_LO"},
	{30, "R_PPC_PLT16_HI"},
	{31, "R_PPC_PLT16_HA"},
	{32, "R_PPC_SDAREL16"},
	{33, "R_PPC_SECTOFF"},
	{34, "R_PPC_SECTOFF_LO"},
	{35, "R_PPC_SECTOFF_HI"},
	{36, "R_PPC_SECTOFF_HA"},
	{67, "R_PPC_TLS"},
	{68, "R_PPC_DTPMOD32"},
	{69, "R_PPC_TPREL16"},
	{70, "R_PPC_TPREL16_LO"},
	{71, "R_PPC_TPREL16_HI"},
	{72, "R_PPC_TPREL16_HA"},
	{73, "R_PPC_TPREL32"},
	{74, "R_PPC_DTPREL16"},
	{75, "R_PPC_DTPREL16_LO"},
	{76, "R_PPC_DTPREL16_HI"},
	{77, "R_PPC_DTPREL16_HA"},
	{78, "R_PPC_DTPREL32"},
	{79, "R_PPC_GOT_TLSGD16"},
	{80, "R_PPC_GOT_TLSGD16_LO"},
	{81, "R_PPC_GOT_TLSGD16_HI"},
	{82, "R_PPC_GOT_TLSGD16_HA"},
	{83, "R_PPC_GOT_TLSLD16"},
	{84, "R_PPC_GOT_TLSLD16_LO"},
	{85, "R_PPC_GOT_TLSLD16_HI"},
	{86, "R_PPC_GOT_TLSLD16_HA"},
	{87, "R_PPC_GOT_TPREL16"},
	{88, "R_PPC_GOT_TPREL16_LO"},
	{89, "R_PPC_GOT_TPREL16_HI"},
	{90, "R_PPC_GOT_TPREL16_HA"},
	{101, "R_PPC_EMB_NADDR32"},
	{102, "R_PPC_EMB_NADDR16"},
	{103, "R_PPC_EMB_NADDR16_LO"},
	{104, "R_PPC_EMB_NADDR16_HI"},
	{105, "R_PPC_EMB_NADDR16_HA"},
	{106, "R_PPC_EMB_SDAI16"},
	{107, "R_PPC_EMB_SDA2I16"},
	{108, "R_PPC_EMB_SDA2REL"},
	{109, "R_PPC_EMB_SDA21"},
	{110, "R_PPC_EMB_MRKREF"},
	{111, "R_PPC_EMB_RELSEC16"},
	{112, "R_PPC_EMB_RELST_LO"},
	{113, "R_PPC_EMB_RELST_HI"},
	{114, "R_PPC_EMB_RELST_HA"},
	{115, "R_PPC_EMB_BIT_FLD"},
	{116, "R_PPC_EMB_RELSDA"},
}

func (i R_PPC) String() string   { return stringName(uint32(i), rppcStrings, false) }
func (i R_PPC) GoString() string { return stringName(uint32(i), rppcStrings, true) }

// Relocation types for 64-bit PowerPC or Power Architecture processors.
//
// Values that are shared by both R_PPC and R_PPC64 are prefixed with
// R_POWERPC_ in the ELF standard. For the R_PPC64 type, the relevant
// shared relocations have been renamed with the prefix R_PPC64_.
// The original name follows the value in a comment.
type R_PPC64 int

const (
	R_PPC64_NONE               R_PPC64 = 0  // R_POWERPC_NONE
	R_PPC64_ADDR32             R_PPC64 = 1  // R_POWERPC_ADDR32
	R_PPC64_ADDR24             R_PPC64 = 2  // R_POWERPC_ADDR24
	R_PPC64_ADDR16             R_PPC64 = 3  // R_POWERPC_ADDR16
	R_PPC64_ADDR16_LO          R_PPC64 = 4  // R_POWERPC_ADDR16_LO
	R_PPC64_ADDR16_HI          R_PPC64 = 5  // R_POWERPC_ADDR16_HI
	R_PPC64_ADDR16_HA          R_PPC64 = 6  // R_POWERPC_ADDR16_HA
	R_PPC64_ADDR14             R_PPC64 = 7  // R_POWERPC_ADDR14
	R_PPC64_ADDR14_BRTAKEN     R_PPC64 = 8  // R_POWERPC_ADDR14_BRTAKEN
	R_PPC64_ADDR14_BRNTAKEN    R_PPC64 = 9  // R_POWERPC_ADDR14_BRNTAKEN
	R_PPC64_REL24              R_PPC64 = 10 // R_POWERPC_REL24
	R_PPC64_REL14              R_PPC64 = 11 // R_POWERPC_REL14
	R_PPC64_REL14_BRTAKEN      R_PPC64 = 12 // R_POWERPC_REL14_BRTAKEN
	R_PPC64_REL14_BRNTAKEN     R_PPC64 = 13 // R_POWERPC_REL14_BRNTAKEN
	R_PPC64_GOT16              R_PPC64 = 14 // R_POWERPC_GOT16
	R_PPC64_GOT16_LO           R_PPC64 = 15 // R_POWERPC_GOT16_LO
	R_PPC64_GOT16_HI           R_PPC64 = 16 // R_POWERPC_GOT16_HI
	R_PPC64_GOT16_HA           R_PPC64 = 17 // R_POWERPC_GOT16_HA
	R_PPC64_COPY               R_PPC64 = 19 // R_POWERPC_COPY
	R_PPC64_GLOB_DAT           R_PPC64 = 20 // R_POWERPC_GLOB_DAT
	R_PPC64_JMP_SLOT           R_PPC64 = 21 // R_POWERPC_JMP_SLOT
	R_PPC64_RELATIVE           R_PPC64 = 22 // R_POWERPC_RELATIVE
	R_PPC64_UADDR32            R_PPC64 = 24 // R_POWERPC_UADDR32
	R_PPC64_UADDR16            R_PPC64 = 25 // R_POWERPC_UADDR16
	R_PPC64_REL32              R_PPC64 = 26 // R_POWERPC_REL32
	R_PPC64_PLT32              R_PPC64 = 27 // R_POWERPC_PLT32
	R_PPC64_PLTREL32           R_PPC64 = 28 // R_POWERPC_PLTREL32
	R_PPC64_PLT16_LO           R_PPC64 = 29 // R_POWERPC_PLT16_LO
	R_PPC64_PLT16_HI           R_PPC64 = 30 // R_POWERPC_PLT16_HI
	R_PPC64_PLT16_HA           R_PPC64 = 31 // R_POWERPC_PLT16_HA
	R_PPC64_SECTOFF            R_PPC64 = 33 // R_POWERPC_SECTOFF
	R_PPC64_SECTOFF_LO         R_PPC64 = 34 // R_POWERPC_SECTOFF_LO
	R_PPC64_SECTOFF_HI         R_PPC64 = 35 // R_POWERPC_SECTOFF_HI
	R_PPC64_SECTOFF_HA         R_PPC64 = 36 // R_POWERPC_SECTOFF_HA
	R_PPC64_REL30              R_PPC64 = 37 // R_POWERPC_ADDR30
	R_PPC64_ADDR64             R_PPC64 = 38
	R_PPC64_ADDR16_HIGHER      R_PPC64 = 39
	R_PPC64_ADDR16_HIGHERA     R_PPC64 = 40
	R_PPC64_ADDR16_HIGHEST     R_PPC64 = 41
	R_PPC64_ADDR16_HIGHESTA    R_PPC64 = 42
	R_PPC64_UADDR64            R_PPC64 = 43
	R_PPC64_REL64              R_PPC64 = 44
	R_PPC64_PLT64              R_PPC64 = 45
	R_PPC64_PLTREL64           R_PPC64 = 46
	R_PPC64_TOC16              R_PPC64 = 47
	R_PPC64_TOC16_LO           R_PPC64 = 48
	R_PPC64_TOC16_HI           R_PPC64 = 49
	R_PPC64_TOC16_HA           R_PPC64 = 50
	R_PPC64_TOC                R_PPC64 = 51
	R_PPC64_PLTGOT16           R_PPC64 = 52
	R_PPC64_PLTGOT16_LO        R_PPC64 = 53
	R_PPC64_PLTGOT16_HI        R_PPC64 = 54
	R_PPC64_PLTGOT16_HA        R_PPC64 = 55
	R_PPC64_ADDR16_DS          R_PPC64 = 56
	R_PPC64_ADDR16_LO_DS       R_PPC64 = 57
	R_PPC64_GOT16_DS           R_PPC64 = 58
	R_PPC64_GOT16_LO_DS        R_PPC64 = 59
	R_PPC64_PLT16_LO_DS        R_PPC64 = 60
	R_PPC64_SECTOFF_DS         R_PPC64 = 61
	R_PPC64_SECTOFF_LO_DS      R_PPC64 = 62
	R_PPC64_TOC16_DS           R_PPC64 = 63
	R_PPC64_TOC16_LO_DS        R_PPC64 = 64
	R_PPC64_PLTGOT16_DS        R_PPC64 = 65
	R_PPC64_PLTGOT_LO_DS       R_PPC64 = 66
	R_PPC64_TLS                R_PPC64 = 67 // R_POWERPC_TLS
	R_PPC64_DTPMOD64           R_PPC64 = 68 // R_POWERPC_DTPMOD64
	R_PPC64_TPREL16            R_PPC64 = 69 // R_POWERPC_TPREL16
	R_PPC64_TPREL16_LO         R_PPC64 = 70 // R_POWERPC_TPREL16_LO
	R_PPC64_TPREL16_HI         R_PPC64 = 71 // R_POWERPC_TPREL16_HI
	R_PPC64_TPREL16_HA         R_PPC64 = 72 // R_POWERPC_TPREL16_HA
	R_PPC64_TPREL64            R_PPC64 = 73 // R_POWERPC_TPREL64
	R_PPC64_DTPREL16           R_PPC64 = 74 // R_POWERPC_DTPREL16
	R_PPC64_DTPREL16_LO        R_PPC64 = 75 // R_POWERPC_DTPREL16_LO
	R_PPC64_DTPREL16_HI        R_PPC64 = 76 // R_POWERPC_DTPREL16_HI
	R_PPC64_DTPREL16_HA        R_PPC64 = 77 // R_POWERPC_DTPREL16_HA
	R_PPC64_DTPREL64           R_PPC64 = 78 // R_POWERPC_DTPREL64
	R_PPC64_GOT_TLSGD16        R_PPC64 = 79 // R_POWERPC_GOT_TLSGD16
	R_PPC64_GOT_TLSGD16_LO     R_PPC64 = 80 // R_POWERPC_GOT_TLSGD16_LO
	R_PPC64_GOT_TLSGD16_HI     R_PPC64 = 81 // R_POWERPC_GOT_TLSGD16_HI
	R_PPC64_GOT_TLSGD16_HA     R_PPC64 = 82 // R_POWERPC_GOT_TLSGD16_HA
	R_PPC64_GOT_TLSLD16        R_PPC64 = 83 // R_POWERPC_GOT_TLSLD16
	R_PPC64_GOT_TLSLD16_LO     R_PPC64 = 84 // R_POWERPC_GOT_TLSLD16_LO
	R_PPC64_GOT_TLSLD16_HI     R_PPC64 = 85 // R_POWERPC_GOT_TLSLD16_HI
	R_PPC64_GOT_TLSLD16_HA     R_PPC64 = 86 // R_POWERPC_GOT_TLSLD16_HA
	R_PPC64_GOT_TPREL16_DS     R_PPC64 = 87 // R_POWERPC_GOT_TPREL16_DS
	R_PPC64_GOT_TPREL16_LO_DS  R_PPC64 = 88 // R_POWERPC_GOT_TPREL16_LO_DS
	R_PPC64_GOT_TPREL16_HI     R_PPC64 = 89 // R_POWERPC_GOT_TPREL16_HI
	R_PPC64_GOT_TPREL16_HA     R_PPC64 = 90 // R_POWERPC_GOT_TPREL16_HA
	R_PPC64_GOT_DTPREL16_DS    R_PPC64 = 91 // R_POWERPC_GOT_DTPREL16_DS
	R_PPC64_GOT_DTPREL16_LO_DS R_PPC64 = 92 // R_POWERPC_GOT_DTPREL16_LO_DS
	R_PPC64_GOT_DTPREL16_HI    R_PPC64 = 93 // R_POWERPC_GOT_DTPREL16_HI
	R_PPC64_GOT_DTPREL16_HA    R_PPC64 = 94 // R_POWERPC_GOT_DTPREL16_HA
	R_PPC64_TPREL16_DS         R_PPC64 = 95
	R_PPC64_TPREL16_LO_DS      R_PPC64 = 96
	R_PPC64_TPREL16_HIGHER     R_PPC64 = 97
	R_PPC64_TPREL16_HIGHERA    R_PPC64 = 98
	R_PPC64_TPREL16_HIGHEST    R_PPC64 = 99
	R_PPC64_TPREL16_HIGHESTA   R_PPC64 = 100
	R_PPC64_DTPREL16_DS        R_PPC64 = 101
	R_PPC64_DTPREL16_LO_DS     R_PPC64 = 102
	R_PPC64_DTPREL16_HIGHER    R_PPC64 = 103
	R_PPC64_DTPREL16_HIGHERA   R_PPC64 = 104
	R_PPC64_DTPREL16_HIGHEST   R_PPC64 = 105
	R_PPC64_DTPREL16_HIGHESTA  R_PPC64 = 106
	R_PPC64_TLSGD              R_PPC64 = 107
	R_PPC64_TLSLD              R_PPC64 = 108
	R_PPC64_TOCSAVE            R_PPC64 = 109
	R_PPC64_ADDR16_HIGH        R_PPC64 = 110
	R_PPC64_ADDR16_HIGHA       R_PPC64 = 111
	R_PPC64_TPREL16_HIGH       R_PPC64 = 112
	R_PPC64_TPREL16_HIGHA      R_PPC64 = 113
	R_PPC64_DTPREL16_HIGH      R_PPC64 = 114
	R_PPC64_DTPREL16_HIGHA     R_PPC64 = 115
	R_PPC64_REL24_NOTOC        R_PPC64 = 116
	R_PPC64_ADDR64_LOCAL       R_PPC64 = 117
	R_PPC64_ENTRY              R_PPC64 = 118
	R_PPC64_PLTSEQ             R_PPC64 = 119
	R_PPC64_PLTCALL            R_PPC64 = 120
	R_PPC64_PLTSEQ_NOTOC       R_PPC64 = 121
	R_PPC64_PLTCALL_NOTOC      R_PPC64 = 122
	R_PPC64_PCREL_OPT          R_PPC64 = 123
	R_PPC64_REL24_P9NOTOC      R_PPC64 = 124
	R_PPC64_D34                R_PPC64 = 128
	R_PPC64_D34_LO             R_PPC64 = 129
	R_PPC64_D34_HI30           R_PPC64 = 130
	R_PPC64_D34_HA30           R_PPC64 = 131
	R_PPC64_PCREL34            R_PPC64 = 132
	R_PPC64_GOT_PCREL34        R_PPC64 = 133
	R_PPC64_PLT_PCREL34        R_PPC64 = 134
	R_PPC64_PLT_PCREL34_NOTOC  R_PPC64 = 135
	R_PPC64_ADDR16_HIGHER34    R_PPC64 = 136
	R_PPC64_ADDR16_HIGHERA34   R_PPC64 = 137
	R_PPC64_ADDR16_HIGHEST34   R_PPC64 = 138
	R_PPC64_ADDR16_HIGHESTA34  R_PPC64 = 139
	R_PPC64_REL16_HIGHER34     R_PPC64 = 140
	R_PPC64_REL16_HIGHERA34    R_PPC64 = 141
	R_PPC64_REL16_HIGHEST34    R_PPC64 = 142
	R_PPC64_REL16_HIGHESTA34   R_PPC64 = 143
	R_PPC64_D28                R_PPC64 = 144
	R_PPC64_PCREL28            R_PPC64 = 145
	R_PPC64_TPREL34            R_PPC64 = 146
	R_PPC64_DTPREL34           R_PPC64 = 147
	R_PPC64_GOT_TLSGD_PCREL34  R_PPC64 = 148
	R_PPC64_GOT_TLSLD_PCREL34  R_PPC64 = 149
	R_PPC64_GOT_TPREL_PCREL34  R_PPC64 = 150
	R_PPC64_GOT_DTPREL_PCREL34 R_PPC64 = 151
	R_PPC64_REL16_HIGH         R_PPC64 = 240
	R_PPC64_REL16_HIGHA        R_PPC64 = 241
	R_PPC64_REL16_HIGHER       R_PPC64 = 242
	R_PPC64_REL16_HIGHERA      R_PPC64 = 243
	R_PPC64_REL16_HIGHEST      R_PPC64 = 244
	R_PPC64_REL16_HIGHESTA     R_PPC64 = 245
	R_PPC64_REL16DX_HA         R_PPC64 = 246 // R_POWERPC_REL16DX_HA
	R_PPC64_JMP_IREL           R_PPC64 = 247
	R_PPC64_IRELATIVE          R_PPC64 = 248 // R_POWERPC_IRELATIVE
	R_PPC64_REL16              R_PPC64 = 249 // R_POWERPC_REL16
	R_PPC64_REL16_LO           R_PPC64 = 250 // R_POWERPC_REL16_LO
	R_PPC64_REL16_HI           R_PPC64 = 251 // R_POWERPC_REL16_HI
	R_PPC64_REL16_HA           R_PPC64 = 252 // R_POWERPC_REL16_HA
	R_PPC64_GNU_VTINHERIT      R_PPC64 = 253
	R_PPC64_GNU_VTENTRY        R_PPC64 = 254
)

var rppc64Strings = []intName{
	{0, "R_PPC64_NONE"},
	{1, "R_PPC64_ADDR32"},
	{2, "R_PPC64_ADDR24"},
	{3, "R_PPC64_ADDR16"},
	{4, "R_PPC64_ADDR16_LO"},
	{5, "R_PPC64_ADDR16_HI"},
	{6, "R_PPC64_ADDR16_HA"},
	{7, "R_PPC64_ADDR14"},
	{8, "R_PPC64_ADDR14_BRTAKEN"},
	{9, "R_PPC64_ADDR14_BRNTAKEN"},
	{10, "R_PPC64_REL24"},
	{11, "R_PPC64_REL14"},
	{12, "R_PPC64_REL14_BRTAKEN"},
	{13, "R_PPC64_REL14_BRNTAKEN"},
	{14, "R_PPC64_GOT16"},
	{15, "R_PPC64_GOT16_LO"},
	{16, "R_PPC64_GOT16_HI"},
	{17, "R_PPC64_GOT16_HA"},
	{19, "R_PPC64_COPY"},
	{20, "R_PPC64_GLOB_DAT"},
	{21, "R_PPC64_JMP_SLOT"},
	{22, "R_PPC64_RELATIVE"},
	{24, "R_PPC64_UADDR32"},
	{25, "R_PPC64_UADDR16"},
	{26, "R_PPC64_REL32"},
	{27, "R_PPC64_PLT32"},
	{28, "R_PPC64_PLTREL32"},
	{29, "R_PPC64_PLT16_LO"},
	{30, "R_PPC64_PLT16_HI"},
	{31, "R_PPC64_PLT16_HA"},
	{33, "R_PPC64_SECTOFF"},
	{34, "R_PPC64_SECTOFF_LO"},
	{35, "R_PPC64_SECTOFF_HI"},
	{36, "R_PPC64_SECTOFF_HA"},
	{37, "R_PPC64_REL30"},
	{38, "R_PPC64_ADDR64"},
	{39, "R_PPC64_ADDR16_HIGHER"},
	{40, "R_PPC64_ADDR16_HIGHERA"},
	{41, "R_PPC64_ADDR16_HIGHEST"},
	{42, "R_PPC64_ADDR16_HIGHESTA"},
	{43, "R_PPC64_UADDR64"},
	{44, "R_PPC64_REL64"},
	{45, "R_PPC64_PLT64"},
	{46, "R_PPC64_PLTREL64"},
	{47, "R_PPC64_TOC16"},
	{48, "R_PPC64_TOC16_LO"},
	{49, "R_PPC64_TOC16_HI"},
	{50, "R_PPC64_TOC16_HA"},
	{51, "R_PPC64_TOC"},
	{52, "R_PPC64_PLTGOT16"},
	{53, "R_PPC64_PLTGOT16_LO"},
	{54, "R_PPC64_PLTGOT16_HI"},
	{55, "R_PPC64_PLTGOT16_HA"},
	{56, "R_PPC64_ADDR16_DS"},
	{57, "R_PPC64_ADDR16_LO_DS"},
	{58, "R_PPC64_GOT16_DS"},
	{59, "R_PPC64_GOT16_LO_DS"},
	{60, "R_PPC64_PLT16_LO_DS"},
	{61, "R_PPC64_SECTOFF_DS"},
	{62, "R_PPC64_SECTOFF_LO_DS"},
	{63, "R_PPC64_TOC16_DS"},
	{64, "R_PPC64_TOC16_LO_DS"},
	{65, "R_PPC64_PLTGOT16_DS"},
	{66, "R_PPC64_PLTGOT_LO_DS"},
	{67, "R_PPC64_TLS"},
	{68, "R_PPC64_DTPMOD64"},
	{69, "R_PPC64_TPREL16"},
	{70, "R_PPC64_TPREL16_LO"},
	{71, "R_PPC64_TPREL16_HI"},
	{72, "R_PPC64_TPREL16_HA"},
	{73, "R_PPC64_TPREL64"},
	{74, "R_PPC64_DTPREL16"},
	{75, "R_PPC64_DTPREL16_LO"},
	{76, "R_PPC64_DTPREL16_HI"},
	{77, "R_PPC64_DTPREL16_HA"},
	{78, "R_PPC64_DTPREL64"},
	{79, "R_PPC64_GOT_TLSGD16"},
	{80, "R_PPC64_GOT_TLSGD16_LO"},
	{81, "R_PPC64_GOT_TLSGD16_HI"},
	{82, "R_PPC64_GOT_TLSGD16_HA"},
	{83, "R_PPC64_GOT_TLSLD16"},
	{84, "R_PPC64_GOT_TLSLD16_LO"},
	{85, "R_PPC64_GOT_TLSLD16_HI"},
	{86, "R_PPC64_GOT_TLSLD16_HA"},
	{87, "R_PPC64_GOT_TPREL16_DS"},
	{88, "R_PPC64_GOT_TPREL16_LO_DS"},
	{89, "R_PPC64_GOT_TPREL16_HI"},
	{90, "R_PPC64_GOT_TPREL16_HA"},
	{91, "R_PPC64_GOT_DTPREL16_DS"},
	{92, "R_PPC64_GOT_DTPREL16_LO_DS"},
	{93, "R_PPC64_GOT_DTPREL16_HI"},
	{94, "R_PPC64_GOT_DTPREL16_HA"},
	{95, "R_PPC64_TPREL16_DS"},
	{96, "R_PPC64_TPREL16_LO_DS"},
	{97, "R_PPC64_TPREL16_HIGHER"},
	{98, "R_PPC64_TPREL16_HIGHERA"},
	{99, "R_PPC64_TPREL16_HIGHEST"},
	{100, "R_PPC64_TPREL16_HIGHESTA"},
	{101, "R_PPC64_DTPREL16_DS"},
	{102, "R_PPC64_DTPREL16_LO_DS"},
	{103, "R_PPC64_DTPREL16_HIGHER"},
	{104, "R_PPC64_DTPREL16_HIGHERA"},
	{105, "R_PPC64_DTPREL16_HIGHEST"},
	{106, "R_PPC64_DTPREL16_HIGHESTA"},
	{107, "R_PPC64_TLSGD"},
	{108, "R_PPC64_TLSLD"},
	{109, "R_PPC64_TOCSAVE"},
	{110, "R_PPC64_ADDR16_HIGH"},
	{111, "R_PPC64_ADDR16_HIGHA"},
	{112, "R_PPC64_TPREL16_HIGH"},
	{113, "R_PPC64_TPREL16_HIGHA"},
	{114, "R_PPC64_DTPREL16_HIGH"},
	{115, "R_PPC64_DTPREL16_HIGHA"},
	{116, "R_PPC64_REL24_NOTOC"},
	{117, "R_PPC64_ADDR64_LOCAL"},
	{118, "R_PPC64_ENTRY"},
	{119, "R_PPC64_PLTSEQ"},
	{120, "R_PPC64_PLTCALL"},
	{121, "R_PPC64_PLTSEQ_NOTOC"},
	{122, "R_PPC64_PLTCALL_NOTOC"},
	{123, "R_PPC64_PCREL_OPT"},
	{124, "R_PPC64_REL24_P9NOTOC"},
	{128, "R_PPC64_D34"},
	{129, "R_PPC64_D34_LO"},
	{130, "R_PPC64_D34_HI30"},
	{131, "R_PPC64_D34_HA30"},
	{132, "R_PPC64_PCREL34"},
	{133, "R_PPC64_GOT_PCREL34"},
	{134, "R_PPC64_PLT_PCREL34"},
	{135, "R_PPC64_PLT_PCREL34_NOTOC"},
	{136, "R_PPC64_ADDR16_HIGHER34"},
	{137, "R_PPC64_ADDR16_HIGHERA34"},
	{138, "R_PPC64_ADDR16_HIGHEST34"},
	{139, "R_PPC64_ADDR16_HIGHESTA34"},
	{140, "R_PPC64_REL16_HIGHER34"},
	{141, "R_PPC64_REL16_HIGHERA34"},
	{142, "R_PPC64_REL16_HIGHEST34"},
	{143, "R_PPC64_REL16_HIGHESTA34"},
	{144, "R_PPC64_D28"},
	{145, "R_PPC64_PCREL28"},
	{146, "R_PPC64_TPREL34"},
	{147, "R_PPC64_DTPREL34"},
	{148, "R_PPC64_GOT_TLSGD_PCREL34"},
	{149, "R_PPC64_GOT_TLSLD_PCREL34"},
	{150, "R_PPC64_GOT_TPREL_PCREL34"},
	{151, "R_PPC64_GOT_DTPREL_PCREL34"},
	{240, "R_PPC64_REL16_HIGH"},
	{241, "R_PPC64_REL16_HIGHA"},
	{242, "R_PPC64_REL16_HIGHER"},
	{243, "R_PPC64_REL16_HIGHERA"},
	{244, "R_PPC64_REL16_HIGHEST"},
	{245, "R_PPC64_REL16_HIGHESTA"},
	{246, "R_PPC64_REL16DX_HA"},
	{247, "R_PPC64_JMP_IREL"},
	{248, "R_PPC64_IRELATIVE"},
	{249, "R_PPC64_REL16"},
	{250, "R_PPC64_REL16_LO"},
	{251, "R_PPC64_REL16_HI"},
	{252, "R_PPC64_REL16_HA"},
	{253, "R_PPC64_GNU_VTINHERIT"},
	{254, "R_PPC64_GNU_VTENTRY"},
}

func (i R_PPC64) String() string   { return stringName(uint32(i), rppc64Strings, false) }
func (i R_PPC64) GoString() string { return stringName(uint32(i), rppc64Strings, true) }

// Relocation types for RISC-V processors.
type R_RISCV int

const (
	R_RISCV_NONE          R_RISCV = 0  /* No relocation. */
	R_RISCV_32            R_RISCV = 1  /* Add 32 bit zero extended symbol value */
	R_RISCV_64            R_RISCV = 2  /* Add 64 bit symbol value. */
	R_RISCV_RELATIVE      R_RISCV = 3  /* Add load address of shared object. */
	R_RISCV_COPY          R_RISCV = 4  /* Copy data from shared object. */
	R_RISCV_JUMP_SLOT     R_RISCV = 5  /* Set GOT entry to code address. */
	R_RISCV_TLS_DTPMOD32  R_RISCV = 6  /* 32 bit ID of module containing symbol */
	R_RISCV_TLS_DTPMOD64  R_RISCV = 7  /* ID of module containing symbol */
	R_RISCV_TLS_DTPREL32  R_RISCV = 8  /* 32 bit relative offset in TLS block */
	R_RISCV_TLS_DTPREL64  R_RISCV = 9  /* Relative offset in TLS block */
	R_RISCV_TLS_TPREL32   R_RISCV = 10 /* 32 bit relative offset in static TLS block */
	R_RISCV_TLS_TPREL64   R_RISCV = 11 /* Relative offset in static TLS block */
	R_RISCV_BRANCH        R_RISCV = 16 /* PC-relative branch */
	R_RISCV_JAL           R_RISCV = 17 /* PC-relative jump */
	R_RISCV_CALL          R_RISCV = 18 /* PC-relative call */
	R_RISCV_CALL_PLT      R_RISCV = 19 /* PC-relative call (PLT) */
	R_RISCV_GOT_HI20      R_RISCV = 20 /* PC-relative GOT reference */
	R_RISCV_TLS_GOT_HI20  R_RISCV = 21 /* PC-relative TLS IE GOT offset */
	R_RISCV_TLS_GD_HI20   R_RISCV = 22 /* PC-relative TLS GD reference */
	R_RISCV_PCREL_HI20    R_RISCV = 23 /* PC-relative reference */
	R_RISCV_PCREL_LO12_I  R_RISCV = 24 /* PC-relative reference */
	R_RISCV_PCREL_LO12_S  R_RISCV = 25 /* PC-relative reference */
	R_RISCV_HI20          R_RISCV = 26 /* Absolute address */
	R_RISCV_LO12_I        R_RISCV = 27 /* Absolute address */
	R_RISCV_LO12_S        R_RISCV = 28 /* Absolute address */
	R_RISCV_TPREL_HI20    R_RISCV = 29 /* TLS LE thread offset */
	R_RISCV_TPREL_LO12_I  R_RISCV = 30 /* TLS LE thread offset */
	R_RISCV_TPREL_LO12_S  R_RISCV = 31 /* TLS LE thread offset */
	R_RISCV_TPREL_ADD     R_RISCV = 32 /* TLS LE thread usage */
	R_RISCV_ADD8          R_RISCV = 33 /* 8-bit label addition */
	R_RISCV_ADD16         R_RISCV = 34 /* 16-bit label addition */
	R_RISCV_ADD32         R_RISCV = 35 /* 32-bit label addition */
	R_RISCV_ADD64         R_RISCV = 36 /* 64-bit label addition */
	R_RISCV_SUB8          R_RISCV = 37 /* 8-bit label subtraction */
	R_RISCV_SUB16         R_RISCV = 38 /* 16-bit label subtraction */
	R_RISCV_SUB32         R_RISCV = 39 /* 32-bit label subtraction */
	R_RISCV_SUB64         R_RISCV = 40 /* 64-bit label subtraction */
	R_RISCV_GNU_VTINHERIT R_RISCV = 41 /* GNU C++ vtable hierarchy */
	R_RISCV_GNU_VTENTRY   R_RISCV = 42 /* GNU C++ vtable member usage */
	R_RISCV_ALIGN         R_RISCV = 43 /* Alignment statement */
	R_RISCV_RVC_BRANCH    R_RISCV = 44 /* PC-relative branch offset */
	R_RISCV_RVC_JUMP      R_RISCV = 45 /* PC-relative jump offset */
	R_RISCV_RVC_LUI       R_RISCV = 46 /* Absolute address */
	R_RISCV_GPREL_I       R_RISCV = 47 /* GP-relative reference */
	R_RISCV_GPREL_S       R_RISCV = 48 /* GP-relative reference */
	R_RISCV_TPREL_I       R_RISCV = 49 /* TP-relative TLS LE load */
	R_RISCV_TPREL_S       R_RISCV = 50 /* TP-relative TLS LE store */
	R_RISCV_RELAX         R_RISCV = 51 /* Instruction pair can be relaxed */
	R_RISCV_SUB6          R_RISCV = 52 /* Local label subtraction */
	R_RISCV_SET6          R_RISCV = 53 /* Local label subtraction */
	R_RISCV_SET8          R_RISCV = 54 /* Local label subtraction */
	R_RISCV_SET16         R_RISCV = 55 /* Local label subtraction */
	R_RISCV_SET32         R_RISCV = 56 /* Local label subtraction */
	R_RISCV_32_PCREL      R_RISCV = 57 /* 32-bit PC relative */
)

var rriscvStrings = []intName{
	{0, "R_RISCV_NONE"},
	{1, "R_RISCV_32"},
	{2, "R_RISCV_64"},
	{3, "R_RISCV_RELATIVE"},
	{4, "R_RISCV_COPY"},
	{5, "R_RISCV_JUMP_SLOT"},
	{6, "R_RISCV_TLS_DTPMOD32"},
	{7, "R_RISCV_TLS_DTPMOD64"},
	{8, "R_RISCV_TLS_DTPREL32"},
	{9, "R_RISCV_TLS_DTPREL64"},
	{10, "R_RISCV_TLS_TPREL32"},
	{11, "R_RISCV_TLS_TPREL64"},
	{16, "R_RISCV_BRANCH"},
	{17, "R_RISCV_JAL"},
	{18, "R_RISCV_CALL"},
	{19, "R_RISCV_CALL_PLT"},
	{20, "R_RISCV_GOT_HI20"},
	{21, "R_RISCV_TLS_GOT_HI20"},
	{22, "R_RISCV_TLS_GD_HI20"},
	{23, "R_RISCV_PCREL_HI20"},
	{24, "R_RISCV_PCREL_LO12_I"},
	{25, "R_RISCV_PCREL_LO12_S"},
	{26, "R_RISCV_HI20"},
	{27, "R_RISCV_LO12_I"},
	{28, "R_RISCV_LO12_S"},
	{29, "R_RISCV_TPREL_HI20"},
	{30, "R_RISCV_TPREL_LO12_I"},
	{31, "R_RISCV_TPREL_LO12_S"},
	{32, "R_RISCV_TPREL_ADD"},
	{33, "R_RISCV_ADD8"},
	{34, "R_RISCV_ADD16"},
	{35, "R_RISCV_ADD32"},
	{36, "R_RISCV_ADD64"},
	{37, "R_RISCV_SUB8"},
	{38, "R_RISCV_SUB16"},
	{39, "R_RISCV_SUB32"},
	{40, "R_RISCV_SUB64"},
	{41, "R_RISCV_GNU_VTINHERIT"},
	{42, "R_RISCV_GNU_VTENTRY"},
	{43, "R_RISCV_ALIGN"},
	{44, "R_RISCV_RVC_BRANCH"},
	{45, "R_RISCV_RVC_JUMP"},
	{46, "R_RISCV_RVC_LUI"},
	{47, "R_RISCV_GPREL_I"},
	{48, "R_RISCV_GPREL_S"},
	{49, "R_RISCV_TPREL_I"},
	{50, "R_RISCV_TPREL_S"},
	{51, "R_RISCV_RELAX"},
	{52, "R_RISCV_SUB6"},
	{53, "R_RISCV_SET6"},
	{54, "R_RISCV_SET8"},
	{55, "R_RISCV_SET16"},
	{56, "R_RISCV_SET32"},
	{57, "R_RISCV_32_PCREL"},
}

func (i R_RISCV) String() string   { return stringName(uint32(i), rriscvStrings, false) }
func (i R_RISCV) GoString() string { return stringName(uint32(i), rriscvStrings, true) }

// Relocation types for s390x processors.
type R_390 int

const (
	R_390_NONE        R_390 = 0
	R_390_8           R_390 = 1
	R_390_12          R_390 = 2
	R_390_16          R_390 = 3
	R_390_32          R_390 = 4
	R_390_PC32        R_390 = 5
	R_390_GOT12       R_390 = 6
	R_390_GOT32       R_390 = 7
	R_390_PLT32       R_390 = 8
	R_390_COPY        R_390 = 9
	R_390_GLOB_DAT    R_390 = 10
	R_390_JMP_SLOT    R_390 = 11
	R_390_RELATIVE    R_390 = 12
	R_390_GOTOFF      R_390 = 13
	R_390_GOTPC       R_390 = 14
	R_390_GOT16       R_390 = 15
	R_390_PC16        R_390 = 16
	R_390_PC16DBL     R_390 = 17
	R_390_PLT16DBL    R_390 = 18
	R_390_PC32DBL     R_390 = 19
	R_390_PLT32DBL    R_390 = 20
	R_390_GOTPCDBL    R_390 = 21
	R_390_64          R_390 = 22
	R_390_PC64        R_390 = 23
	R_390_GOT64       R_390 = 24
	R_390_PLT64       R_390 = 25
	R_390_GOTENT      R_390 = 26
	R_390_GOTOFF16    R_390 = 27
	R_390_GOTOFF64    R_390 = 28
	R_390_GOTPLT12    R_390 = 29
	R_390_GOTPLT16    R_390 = 30
	R_390_GOTPLT32    R_390 = 31
	R_390_GOTPLT64    R_390 = 32
	R_390_GOTPLTENT   R_390 = 33
	R_390_GOTPLTOFF16 R_390 = 34
	R_390_GOTPLTOFF32 R_390 = 35
	R_390_GOTPLTOFF64 R_390 = 36
	R_390_TLS_LOAD    R_390 = 37
	R_390_TLS_GDCALL  R_390 = 38
	R_390_TLS_LDCALL  R_390 = 39
	R_390_TLS_GD32    R_390 = 40
	R_390_TLS_GD64    R_390 = 41
	R_390_TLS_GOTIE12 R_390 = 42
	R_390_TLS_GOTIE32 R_390 = 43
	R_390_TLS_GOTIE64 R_390 = 44
	R_390_TLS_LDM32   R_390 = 45
	R_390_TLS_LDM64   R_390 = 46
	R_390_TLS_IE32    R_390 = 47
	R_390_TLS_IE64    R_390 = 48
	R_390_TLS_IEENT   R_390 = 49
	R_390_TLS_LE32    R_390 = 50
	R_390_TLS_LE64    R_390 = 51
	R_390_TLS_LDO32   R_390 = 52
	R_390_TLS_LDO64   R_390 = 53
	R_390_TLS_DTPMOD  R_390 = 54
	R_390_TLS_DTPOFF  R_390 = 55
	R_390_TLS_TPOFF   R_390 = 56
	R_390_20          R_390 = 57
	R_390_GOT20       R_390 = 58
	R_390_GOTPLT20    R_390 = 59
	R_390_TLS_GOTIE20 R_390 = 60
)

var r390Strings = []intName{
	{0, "R_390_NONE"},
	{1, "R_390_8"},
	{2, "R_390_12"},
	{3, "R_390_16"},
	{4, "R_390_32"},
	{5, "R_390_PC32"},
	{6, "R_390_GOT12"},
	{7, "R_390_GOT32"},
	{8, "R_390_PLT32"},
	{9, "R_390_COPY"},
	{10, "R_390_GLOB_DAT"},
	{11, "R_390_JMP_SLOT"},
	{12, "R_390_RELATIVE"},
	{13, "R_390_GOTOFF"},
	{14, "R_390_GOTPC"},
	{15, "R_390_GOT16"},
	{16, "R_390_PC16"},
	{17, "R_390_PC16DBL"},
	{18, "R_390_PLT16DBL"},
	{19, "R_390_PC32DBL"},
	{20, "R_390_PLT32DBL"},
	{21, "R_390_GOTPCDBL"},
	{22, "R_390_64"},
	{23, "R_390_PC64"},
	{24, "R_390_GOT64"},
	{25, "R_390_PLT64"},
	{26, "R_390_GOTENT"},
	{27, "R_390_GOTOFF16"},
	{28, "R_390_GOTOFF64"},
	{29, "R_390_GOTPLT12"},
	{30, "R_390_GOTPLT16"},
	{31, "R_390_GOTPLT32"},
	{32, "R_390_GOTPLT64"},
	{33, "R_390_GOTPLTENT"},
	{34, "R_390_GOTPLTOFF16"},
	{35, "R_390_GOTPLTOFF32"},
	{36, "R_390_GOTPLTOFF64"},
	{37, "R_390_TLS_LOAD"},
	{38, "R_390_TLS_GDCALL"},
	{39, "R_390_TLS_LDCALL"},
	{40, "R_390_TLS_GD32"},
	{41, "R_390_TLS_GD64"},
	{42, "R_390_TLS_GOTIE12"},
	{43, "R_390_TLS_GOTIE32"},
	{44, "R_390_TLS_GOTIE64"},
	{45, "R_390_TLS_LDM32"},
	{46, "R_390_TLS_LDM64"},
	{47, "R_390_TLS_IE32"},
	{48, "R_390_TLS_IE64"},
	{49, "R_390_TLS_IEENT"},
	{50, "R_390_TLS_LE32"},
	{51, "R_390_TLS_LE64"},
	{52, "R_390_TLS_LDO32"},
	{53, "R_390_TLS_LDO64"},
	{54, "R_390_TLS_DTPMOD"},
	{55, "R_390_TLS_DTPOFF"},
	{56, "R_390_TLS_TPOFF"},
	{57, "R_390_20"},
	{58, "R_390_GOT20"},
	{59, "R_390_GOTPLT20"},
	{60, "R_390_TLS_GOTIE20"},
}

func (i R_390) String() string   { return stringName(uint32(i), r390Strings, false) }
func (i R_390) GoString() string { return stringName(uint32(i), r390Strings, true) }

// Relocation types for SPARC.
type R_SPARC int

const (
	R_SPARC_NONE     R_SPARC = 0
	R_SPARC_8        R_SPARC = 1
	R_SPARC_16       R_SPARC = 2
	R_SPARC_32       R_SPARC = 3
	R_SPARC_DISP8    R_SPARC = 4
	R_SPARC_DISP16   R_SPARC = 5
	R_SPARC_DISP32   R_SPARC = 6
	R_SPARC_WDISP30  R_SPARC = 7
	R_SPARC_WDISP22  R_SPARC = 8
	R_SPARC_HI22     R_SPARC = 9
	R_SPARC_22       R_SPARC = 10
	R_SPARC_13       R_SPARC = 11
	R_SPARC_LO10     R_SPARC = 12
	R_SPARC_GOT10    R_SPARC = 13
	R_SPARC_GOT13    R_SPARC = 14
	R_SPARC_GOT22    R_SPARC = 15
	R_SPARC_PC10     R_SPARC = 16
	R_SPARC_PC22     R_SPARC = 17
	R_SPARC_WPLT30   R_SPARC = 18
	R_SPARC_COPY     R_SPARC = 19
	R_SPARC_GLOB_DAT R_SPARC = 20
	R_SPARC_JMP_SLOT R_SPARC = 21
	R_SPARC_RELATIVE R_SPARC = 22
	R_SPARC_UA32     R_SPARC = 23
	R_SPARC_PLT32    R_SPARC = 24
	R_SPARC_HIPLT22  R_SPARC = 25
	R_SPARC_LOPLT10  R_SPARC = 26
	R_SPARC_PCPLT32  R_SPARC = 27
	R_SPARC_PCPLT22  R_SPARC = 28
	R_SPARC_PCPLT10  R_SPARC = 29
	R_SPARC_10       R_SPARC = 30
	R_SPARC_11       R_SPARC = 31
	R_SPARC_64       R_SPARC = 32
	R_SPARC_OLO10    R_SPARC = 33
	R_SPARC_HH22     R_SPARC = 34
	R_SPARC_HM10     R_SPARC = 35
	R_SPARC_LM22     R_SPARC = 36
	R_SPARC_PC_HH22  R_SPARC = 37
	R_SPARC_PC_HM10  R_SPARC = 38
	R_SPARC_PC_LM22  R_SPARC = 39
	R_SPARC_WDISP16  R_SPARC = 40
	R_SPARC_WDISP19  R_SPARC = 41
	R_SPARC_GLOB_JMP R_SPARC = 42
	R_SPARC_7        R_SPARC = 43
	R_SPARC_5        R_SPARC = 44
	R_SPARC_6        R_SPARC = 45
	R_SPARC_DISP64   R_SPARC = 46
	R_SPARC_PLT64    R_SPARC = 47
	R_SPARC_HIX22    R_SPARC = 48
	R_SPARC_LOX10    R_SPARC = 49
	R_SPARC_H44      R_SPARC = 50
	R_SPARC_M44      R_SPARC = 51
	R_SPARC_L44      R_SPARC = 52
	R_SPARC_REGISTER R_SPARC = 53
	R_SPARC_UA64     R_SPARC = 54
	R_SPARC_UA16     R_SPARC = 55
)

var rsparcStrings = []intName{
	{0, "R_SPARC_NONE"},
	{1, "R_SPARC_8"},
	{2, "R_SPARC_16"},
	{3, "R_SPARC_32"},
	{4, "R_SPARC_DISP8"},
	{5, "R_SPARC_DISP16"},
	{6, "R_SPARC_DISP32"},
	{7, "R_SPARC_WDISP30"},
	{8, "R_SPARC_WDISP22"},
	{9, "R_SPARC_HI22"},
	{10, "R_SPARC_22"},
	{11, "R_SPARC_13"},
	{12, "R_SPARC_LO10"},
	{13, "R_SPARC_GOT10"},
	{14, "R_SPARC_GOT13"},
	{15, "R_SPARC_GOT22"},
	{16, "R_SPARC_PC10"},
	{17, "R_SPARC_PC22"},
	{18, "R_SPARC_WPLT30"},
	{19, "R_SPARC_COPY"},
	{20, "R_SPARC_GLOB_DAT"},
	{21, "R_SPARC_JMP_SLOT"},
	{22, "R_SPARC_RELATIVE"},
	{23, "R_SPARC_UA32"},
	{24, "R_SPARC_PLT32"},
	{25, "R_SPARC_HIPLT22"},
	{26, "R_SPARC_LOPLT10"},
	{27, "R_SPARC_PCPLT32"},
	{28, "R_SPARC_PCPLT22"},
	{29, "R_SPARC_PCPLT10"},
	{30, "R_SPARC_10"},
	{31, "R_SPARC_11"},
	{32, "R_SPARC_64"},
	{33, "R_SPARC_OLO10"},
	{34, "R_SPARC_HH22"},
	{35, "R_SPARC_HM10"},
	{36, "R_SPARC_LM22"},
	{37, "R_SPARC_PC_HH22"},
	{38, "R_SPARC_PC_HM10"},
	{39, "R_SPARC_PC_LM22"},
	{40, "R_SPARC_WDISP16"},
	{41, "R_SPARC_WDISP19"},
	{42, "R_SPARC_GLOB_JMP"},
	{43, "R_SPARC_7"},
	{44, "R_SPARC_5"},
	{45, "R_SPARC_6"},
	{46, "R_SPARC_DISP64"},
	{47, "R_SPARC_PLT64"},
	{48, "R_SPARC_HIX22"},
	{49, "R_SPARC_LOX10"},
	{50, "R_SPARC_H44"},
	{51, "R_SPARC_M44"},
	{52, "R_SPARC_L44"},
	{53, "R_SPARC_REGISTER"},
	{54, "R_SPARC_UA64"},
	{55, "R_SPARC_UA16"},
}

func (i R_SPARC) String() string   { return stringName(uint32(i), rsparcStrings, false) }
func (i R_SPARC) GoString() string { return stringName(uint32(i), rsparcStrings, true) }

// Magic number for the elf trampoline, chosen wisely to be an immediate value.
const ARM_MAGIC_TRAMP_NUMBER = 0x5c000003

// ELF32 File header.
type Header32 struct {
	Ident     [EI_NIDENT]byte /* File identification. */
	Type      uint16          /* File type. */
	Machine   uint16          /* Machine architecture. */
	Version   uint32          /* ELF format version. */
	Entry     uint32          /* Entry point. */
	Phoff     uint32          /* Program header file offset. */
	Shoff     uint32          /* Section header file offset. */
	Flags     uint32          /* Architecture-specific flags. */
	Ehsize    uint16          /* Size of ELF header in bytes. */
	Phentsize uint16          /* Size of program header entry. */
	Phnum     uint16          /* Number of program header entries. */
	Shentsize uint16          /* Size of section header entry. */
	Shnum     uint16          /* Number of section header entries. */
	Shstrndx  uint16          /* Section name strings section. */
}

// ELF32 Section header.
type Section32 struct {
	Name      uint32 /* Section name (index into the section header string table). */
	Type      uint32 /* Section type. */
	Flags     uint32 /* Section flags. */
	Addr      uint32 /* Address in memory image. */
	Off       uint32 /* Offset in file. */
	Size      uint32 /* Size in bytes. */
	Link      uint32 /* Index of a related section. */
	Info      uint32 /* Depends on section type. */
	Addralign uint32 /* Alignment in bytes. */
	Entsize   uint32 /* Size of each entry in section. */
}

// ELF32 Program header.
type Prog32 struct {
	Type   uint32 /* Entry type. */
	Off    uint32 /* File offset of contents. */
	Vaddr  uint32 /* Virtual address in memory image. */
	Paddr  uint32 /* Physical address (not used). */
	Filesz uint32 /* Size of contents in file. */
	Memsz  uint32 /* Size of contents in memory. */
	Flags  uint32 /* Access permission flags. */
	Align  uint32 /* Alignment in memory and file. */
}

// ELF32 Dynamic structure. The ".dynamic" section contains an array of them.
type Dyn32 struct {
	Tag int32  /* Entry type. */
	Val uint32 /* Integer/Address value. */
}

// ELF32 Compression header.
type Chdr32 struct {
	Type      uint32
	Size      uint32
	Addralign uint32
}

/*
 * Relocation entries.
 */

// ELF32 Relocations that don't need an addend field.
type Rel32 struct {
	Off  uint32 /* Location to be relocated. */
	Info uint32 /* Relocation type and symbol index. */
}

// ELF32 Relocations that need an addend field.
type Rela32 struct {
	Off    uint32 /* Location to be relocated. */
	Info   uint32 /* Relocation type and symbol index. */
	Addend int32  /* Addend. */
}

func R_SYM32(info uint32) uint32      { return info >> 8 }
func R_TYPE32(info uint32) uint32     { return info & 0xff }
func R_INFO32(sym, typ uint32) uint32 { return sym<<8 | typ }

// ELF32 Symbol.
type Sym32 struct {
	Name  uint32
	Value uint32
	Size  uint32
	Info  uint8
	Other uint8
	Shndx uint16
}

const Sym32Size = 16

func ST_BIND(info uint8) SymBind { return SymBind(info >> 4) }
func ST_TYPE(info uint8) SymType { return SymType(info & 0xF) }
func ST_INFO(bind SymBind, typ SymType) uint8 {
	return uint8(bind)<<4 | uint8(typ)&0xf
}
func ST_VISIBILITY(other uint8) SymVis { return SymVis(other & 3) }

/*
 * ELF64
 */

// ELF64 file header.
type Header64 struct {
	Ident     [EI_NIDENT]byte /* File identification. */
	Type      uint16          /* File type. */
	Machine   uint16          /* Machine architecture. */
	Version   uint32          /* ELF format version. */
	Entry     uint64          /* Entry point. */
	Phoff     uint64          /* Program header file offset. */
	Shoff     uint64          /* Section header file offset. */
	Flags     uint32          /* Architecture-specific flags. */
	Ehsize    uint16          /* Size of ELF header in bytes. */
	Phentsize uint16          /* Size of program header entry. */
	Phnum     uint16          /* Number of program header entries. */
	Shentsize uint16          /* Size of section header entry. */
	Shnum     uint16          /* Number of section header entries. */
	Shstrndx  uint16          /* Section name strings section. */
}

// ELF64 Section header.
type Section64 struct {
	Name      uint32 /* Section name (index into the section header string table). */
	Type      uint32 /* Section type. */
	Flags     uint64 /* Section flags. */
	Addr      uint64 /* Address in memory image. */
	Off       uint64 /* Offset in file. */
	Size      uint64 /* Size in bytes. */
	Link      uint32 /* Index of a related section. */
	Info      uint32 /* Depends on section type. */
	Addralign uint64 /* Alignment in bytes. */
	Entsize   uint64 /* Size of each entry in section. */
}

// ELF64 Program header.
type Prog64 struct {
	Type   uint32 /* Entry type. */
	Flags  uint32 /* Access permission flags. */
	Off    uint64 /* File offset of contents. */
	Vaddr  uint64 /* Virtual address in memory image. */
	Paddr  uint64 /* Physical address (not used). */
	Filesz uint64 /* Size of contents in file. */
	Memsz  uint64 /* Size of contents in memory. */
	Align  uint64 /* Alignment in memory and file. */
}

// ELF64 Dynamic structure. The ".dynamic" section contains an array of them.
type Dyn64 struct {
	Tag int64  /* Entry type. */
	Val uint64 /* Integer/address value */
}

// ELF64 Compression header.
type Chdr64 struct {
	Type      uint32
	_         uint32 /* Reserved. */
	Size      uint64
	Addralign uint64
}

/*
 * Relocation entries.
 */

/* ELF64 relocations that don't need an addend field. */
type Rel64 struct {
	Off  uint64 /* Location to be relocated. */
	Info uint64 /* Relocation type and symbol index. */
}

/* ELF64 relocations that need an addend field. */
type Rela64 struct {
	Off    uint64 /* Location to be relocated. */
	Info   uint64 /* Relocation type and symbol index. */
	Addend int64  /* Addend. */
}

func R_SYM64(info uint64) uint32    { return uint32(info >> 32) }
func R_TYPE64(info uint64) uint32   { return uint32(info) }
func R_INFO(sym, typ uint32) uint64 { return uint64(sym)<<32 | uint64(typ) }

// ELF64 symbol table entries.
type Sym64 struct {
	Name  uint32 /* String table index of name. */
	Info  uint8  /* Type and binding information. */
	Other uint8  /* Reserved (not used). */
	Shndx uint16 /* Section index of symbol. */
	Value uint64 /* Symbol value. */
	Size  uint64 /* Size of associated object. */
}

const Sym64Size = 24

type intName struct {
	i uint32
	s string
}

func stringName(i uint32, names []intName, goSyntax bool) string {
	for _, n := range names {
		if n.i == i {
			if goSyntax {
				return "elf." + n.s
			}
			return n.s
		}
	}

	// second pass - look for smaller to add with.
	// assume sorted already
	for j := len(names) - 1; j >= 0; j-- {
		n := names[j]
		if n.i < i {
			s := n.s
			if goSyntax {
				s = "elf." + s
			}
			return s + "+" + strconv.FormatUint(uint64(i-n.i), 10)
		}
	}

	return strconv.FormatUint(uint64(i), 10)
}

func flagName(i uint32, names []intName, goSyntax bool) string {
	s := ""
	for _, n := range names {
		if n.i&i == n.i {
			if len(s) > 0 {
				s += "+"
			}
			if goSyntax {
				s += "elf."
			}
			s += n.s
			i -= n.i
		}
	}
	if len(s) == 0 {
		return "0x" + strconv.FormatUint(uint64(i), 16)
	}
	if i != 0 {
		s += "+0x" + strconv.FormatUint(uint64(i), 16)
	}
	return s
}
