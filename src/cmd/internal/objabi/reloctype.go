// Derived from Inferno utils/6l/l.h and related files.
// https://bitbucket.org/inferno-os/inferno-os/src/master/utils/6l/l.h
//
//	Copyright © 1994-1999 Lucent Technologies Inc.  All rights reserved.
//	Portions Copyright © 1995-1997 C H Forsyth (forsyth@terzarima.net)
//	Portions Copyright © 1997-1999 Vita Nuova Limited
//	Portions Copyright © 2000-2007 Vita Nuova Holdings Limited (www.vitanuova.com)
//	Portions Copyright © 2004,2006 Bruce Ellis
//	Portions Copyright © 2005-2007 C H Forsyth (forsyth@terzarima.net)
//	Revisions Copyright © 2000-2007 Lucent Technologies Inc. and others
//	Portions Copyright © 2009 The Go Authors. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.  IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.

package objabi

type RelocType int16

//go:generate stringer -type=RelocType
const (
	R_ADDR RelocType = 1 + iota
	// R_ADDRPOWER relocates a pair of "D-form" instructions (instructions with 16-bit
	// immediates in the low half of the instruction word), usually addis followed by
	// another add or a load, inserting the "high adjusted" 16 bits of the address of
	// the referenced symbol into the immediate field of the first instruction and the
	// low 16 bits into that of the second instruction.
	R_ADDRPOWER
	// R_ADDRARM64 relocates an adrp, add pair to compute the address of the
	// referenced symbol.
	R_ADDRARM64
	// R_ADDRMIPS (only used on mips/mips64) resolves to the low 16 bits of an external
	// address, by encoding it into the instruction.
	R_ADDRMIPS
	// R_ADDROFF resolves to a 32-bit offset from the beginning of the section
	// holding the data being relocated to the referenced symbol.
	R_ADDROFF
	R_SIZE
	R_CALL
	R_CALLARM
	R_CALLARM64
	R_CALLIND
	R_CALLPOWER
	// R_CALLMIPS (only used on mips64) resolves to non-PC-relative target address
	// of a CALL (JAL) instruction, by encoding the address into the instruction.
	R_CALLMIPS
	R_CONST
	R_PCREL
	// R_TLS_LE, used on 386, amd64, and ARM, resolves to the offset of the
	// thread-local symbol from the thread local base and is used to implement the
	// "local exec" model for tls access (r.Sym is not set on intel platforms but is
	// set to a TLS symbol -- runtime.tlsg -- in the linker when externally linking).
	R_TLS_LE
	// R_TLS_IE, used 386, amd64, and ARM resolves to the PC-relative offset to a GOT
	// slot containing the offset from the thread-local symbol from the thread local
	// base and is used to implemented the "initial exec" model for tls access (r.Sym
	// is not set on intel platforms but is set to a TLS symbol -- runtime.tlsg -- in
	// the linker when externally linking).
	R_TLS_IE
	R_GOTOFF
	R_PLT0
	R_PLT1
	R_PLT2
	R_USEFIELD
	// R_USETYPE resolves to an *rtype, but no relocation is created. The
	// linker uses this as a signal that the pointed-to type information
	// should be linked into the final binary, even if there are no other
	// direct references. (This is used for types reachable by reflection.)
	R_USETYPE
	// R_USEIFACE marks a type is converted to an interface in the function this
	// relocation is applied to. The target is a type descriptor or an itab
	// (in the latter case it refers to the concrete type contained in the itab).
	// This is a marker relocation (0-sized), for the linker's reachabililty
	// analysis.
	R_USEIFACE
	// R_USEIFACEMETHOD marks an interface method that is used in the function
	// this relocation is applied to. The target is an interface type descriptor.
	// The addend is the offset of the method in the type descriptor.
	// This is a marker relocation (0-sized), for the linker's reachabililty
	// analysis.
	R_USEIFACEMETHOD
	// R_USENAMEDMETHOD marks that methods with a specific name must not be eliminated.
	// The target is a symbol containing the name of a method called via a generic
	// interface or looked up via MethodByName("F").
	R_USENAMEDMETHOD
	// R_METHODOFF resolves to a 32-bit offset from the beginning of the section
	// holding the data being relocated to the referenced symbol.
	// It is a variant of R_ADDROFF used when linking from the uncommonType of a
	// *rtype, and may be set to zero by the linker if it determines the method
	// text is unreachable by the linked program.
	R_METHODOFF
	// R_KEEP tells the linker to keep the referred-to symbol in the final binary
	// if the symbol containing the R_KEEP relocation is in the final binary.
	R_KEEP
	R_POWER_TOC
	R_GOTPCREL
	// R_JMPMIPS (only used on mips64) resolves to non-PC-relative target address
	// of a JMP instruction, by encoding the address into the instruction.
	// The stack nosplit check ignores this since it is not a function call.
	R_JMPMIPS

	// R_DWARFSECREF resolves to the offset of the symbol from its section.
	// Target of relocation must be size 4 (in current implementation).
	R_DWARFSECREF

	// R_DWARFFILEREF resolves to an index into the DWARF .debug_line
	// file table for the specified file symbol. Must be applied to an
	// attribute of form DW_FORM_data4.
	R_DWARFFILEREF

	// Platform dependent relocations. Architectures with fixed width instructions
	// have the inherent issue that a 32-bit (or 64-bit!) displacement cannot be
	// stuffed into a 32-bit instruction, so an address needs to be spread across
	// several instructions, and in turn this requires a sequence of relocations, each
	// updating a part of an instruction. This leads to relocation codes that are
	// inherently processor specific.

	// Arm64.

	// Set a MOV[NZ] immediate field to bits [15:0] of the offset from the thread
	// local base to the thread local variable defined by the referenced (thread
	// local) symbol. Error if the offset does not fit into 16 bits.
	R_ARM64_TLS_LE

	// Relocates an ADRP; LD64 instruction sequence to load the offset between
	// the thread local base and the thread local variable defined by the
	// referenced (thread local) symbol from the GOT.
	R_ARM64_TLS_IE

	// R_ARM64_GOTPCREL relocates an adrp, ld64 pair to compute the address of the GOT
	// slot of the referenced symbol.
	R_ARM64_GOTPCREL

	// R_ARM64_GOT resolves a GOT-relative instruction sequence, usually an adrp
	// followed by another ld instruction.
	R_ARM64_GOT

	// R_ARM64_PCREL resolves a PC-relative addresses instruction sequence, usually an
	// adrp followed by another add instruction.
	R_ARM64_PCREL

	// R_ARM64_PCREL_LDST8 resolves a PC-relative addresses instruction sequence, usually an
	// adrp followed by a LD8 or ST8 instruction.
	R_ARM64_PCREL_LDST8

	// R_ARM64_PCREL_LDST16 resolves a PC-relative addresses instruction sequence, usually an
	// adrp followed by a LD16 or ST16 instruction.
	R_ARM64_PCREL_LDST16

	// R_ARM64_PCREL_LDST32 resolves a PC-relative addresses instruction sequence, usually an
	// adrp followed by a LD32 or ST32 instruction.
	R_ARM64_PCREL_LDST32

	// R_ARM64_PCREL_LDST64 resolves a PC-relative addresses instruction sequence, usually an
	// adrp followed by a LD64 or ST64 instruction.
	R_ARM64_PCREL_LDST64

	// R_ARM64_LDST8 sets a LD/ST immediate value to bits [11:0] of a local address.
	R_ARM64_LDST8

	// R_ARM64_LDST16 sets a LD/ST immediate value to bits [11:1] of a local address.
	R_ARM64_LDST16

	// R_ARM64_LDST32 sets a LD/ST immediate value to bits [11:2] of a local address.
	R_ARM64_LDST32

	// R_ARM64_LDST64 sets a LD/ST immediate value to bits [11:3] of a local address.
	R_ARM64_LDST64

	// R_ARM64_LDST128 sets a LD/ST immediate value to bits [11:4] of a local address.
	R_ARM64_LDST128

	// PPC64.

	// R_POWER_TLS_LE is used to implement the "local exec" model for tls
	// access. It resolves to the offset of the thread-local symbol from the
	// thread pointer (R13) and is split against a pair of instructions to
	// support a 32 bit displacement.
	R_POWER_TLS_LE

	// R_POWER_TLS_IE is used to implement the "initial exec" model for tls access. It
	// relocates a D-form, DS-form instruction sequence like R_ADDRPOWER_DS. It
	// inserts to the offset of GOT slot for the thread-local symbol from the TOC (the
	// GOT slot is filled by the dynamic linker with the offset of the thread-local
	// symbol from the thread pointer (R13)).
	R_POWER_TLS_IE

	// R_POWER_TLS marks an X-form instruction such as "ADD R3,R13,R4" as completing
	// a sequence of GOT-relative relocations to compute a TLS address. This can be
	// used by the system linker to to rewrite the GOT-relative TLS relocation into a
	// simpler thread-pointer relative relocation. See table 3.26 and 3.28 in the
	// ppc64 elfv2 1.4 ABI on this transformation.  Likewise, the second argument
	// (usually called RB in X-form instructions) is assumed to be R13.
	R_POWER_TLS

	// R_POWER_TLS_IE_PCREL34 is similar to R_POWER_TLS_IE, but marks a single MOVD
	// which has been assembled as a single prefixed load doubleword without using the
	// TOC.
	R_POWER_TLS_IE_PCREL34

	// R_POWER_TLS_LE_TPREL34 is similar to R_POWER_TLS_LE, but computes an offset from
	// the thread pointer in one prefixed instruction.
	R_POWER_TLS_LE_TPREL34

	// R_ADDRPOWER_DS is similar to R_ADDRPOWER above, but assumes the second
	// instruction is a "DS-form" instruction, which has an immediate field occupying
	// bits [15:2] of the instruction word. Bits [15:2] of the address of the
	// relocated symbol are inserted into this field; it is an error if the last two
	// bits of the address are not 0.
	R_ADDRPOWER_DS

	// R_ADDRPOWER_GOT relocates a D-form + DS-form instruction sequence by inserting
	// a relative displacement of referenced symbol's GOT entry to the TOC pointer.
	R_ADDRPOWER_GOT

	// R_ADDRPOWER_GOT_PCREL34 is identical to R_ADDRPOWER_GOT, but uses a PC relative
	// sequence to generate a GOT symbol addresses.
	R_ADDRPOWER_GOT_PCREL34

	// R_ADDRPOWER_PCREL relocates two D-form instructions like R_ADDRPOWER, but
	// inserts the displacement from the place being relocated to the address of the
	// relocated symbol instead of just its address.
	R_ADDRPOWER_PCREL

	// R_ADDRPOWER_TOCREL relocates two D-form instructions like R_ADDRPOWER, but
	// inserts the offset from the TOC to the address of the relocated symbol
	// rather than the symbol's address.
	R_ADDRPOWER_TOCREL

	// R_ADDRPOWER_TOCREL_DS relocates a D-form, DS-form instruction sequence like
	// R_ADDRPOWER_DS but inserts the offset from the TOC to the address of the
	// relocated symbol rather than the symbol's address.
	R_ADDRPOWER_TOCREL_DS

	// R_ADDRPOWER_D34 relocates a single prefixed D-form load/store operation.  All
	// prefixed forms are D form. The high 18 bits are stored in the prefix,
	// and the low 16 are stored in the suffix. The address is absolute.
	R_ADDRPOWER_D34

	// R_ADDRPOWER_PCREL34 relates a single prefixed D-form load/store/add operation.
	// All prefixed forms are D form. The resulting address is relative to the
	// PC. It is a signed 34 bit offset.
	R_ADDRPOWER_PCREL34

	// RISC-V.

	// R_RISCV_JAL resolves a 20 bit offset for a J-type instruction.
	R_RISCV_JAL

	// R_RISCV_JAL_TRAMP is the same as R_RISCV_JAL but denotes the use of a
	// trampoline, which we may be able to avoid during relocation. These are
	// only used by the linker and are not emitted by the compiler or assembler.
	R_RISCV_JAL_TRAMP

	// R_RISCV_CALL resolves a 32 bit PC-relative address for an AUIPC + JALR
	// instruction pair.
	R_RISCV_CALL

	// R_RISCV_PCREL_ITYPE resolves a 32 bit PC-relative address for an
	// AUIPC + I-type instruction pair.
	R_RISCV_PCREL_ITYPE

	// R_RISCV_PCREL_STYPE resolves a 32 bit PC-relative address for an
	// AUIPC + S-type instruction pair.
	R_RISCV_PCREL_STYPE

	// R_RISCV_TLS_IE resolves a 32 bit TLS initial-exec address for an
	// AUIPC + I-type instruction pair.
	R_RISCV_TLS_IE

	// R_RISCV_TLS_LE resolves a 32 bit TLS local-exec address for a
	// LUI + I-type instruction sequence.
	R_RISCV_TLS_LE

	// R_RISCV_GOT_HI20 resolves the high 20 bits of a 32-bit PC-relative GOT
	// address.
	R_RISCV_GOT_HI20

	// R_RISCV_PCREL_HI20 resolves the high 20 bits of a 32-bit PC-relative
	// address.
	R_RISCV_PCREL_HI20

	// R_RISCV_PCREL_LO12_I resolves the low 12 bits of a 32-bit PC-relative
	// address using an I-type instruction.
	R_RISCV_PCREL_LO12_I

	// R_RISCV_PCREL_LO12_S resolves the low 12 bits of a 32-bit PC-relative
	// address using an S-type instruction.
	R_RISCV_PCREL_LO12_S

	// R_RISCV_BRANCH resolves a 12-bit PC-relative branch offset.
	R_RISCV_BRANCH

	// R_RISCV_RVC_BRANCH resolves an 8-bit PC-relative offset for a CB-type
	// instruction.
	R_RISCV_RVC_BRANCH

	// R_RISCV_RVC_JUMP resolves an 11-bit PC-relative offset for a CJ-type
	// instruction.
	R_RISCV_RVC_JUMP

	// R_PCRELDBL relocates s390x 2-byte aligned PC-relative addresses.
	// TODO(mundaym): remove once variants can be serialized - see issue 14218.
	R_PCRELDBL

	// Loong64.

	// R_ADDRLOONG64 resolves to the low 12 bits of an external address, by encoding
	// it into the instruction.
	R_ADDRLOONG64

	// R_ADDRLOONG64U resolves to the sign-adjusted "upper" 20 bits (bit 5-24) of an
	// external address, by encoding it into the instruction.
	R_ADDRLOONG64U

	// R_ADDRLOONG64TLS resolves to the low 12 bits of a TLS address (offset from
	// thread pointer), by encoding it into the instruction.
	R_ADDRLOONG64TLS

	// R_ADDRLOONG64TLSU resolves to the high 20 bits of a TLS address (offset from
	// thread pointer), by encoding it into the instruction.
	R_ADDRLOONG64TLSU

	// R_CALLLOONG64 resolves to non-PC-relative target address of a CALL (BL/JIRL)
	// instruction, by encoding the address into the instruction.
	R_CALLLOONG64

	// R_LOONG64_TLS_IE_PCREL_HI and R_LOONG64_TLS_IE_LO relocates a pcalau12i, ld.d
	// pair to compute the address of the GOT slot of the tls symbol.
	R_LOONG64_TLS_IE_PCREL_HI
	R_LOONG64_TLS_IE_LO

	// R_LOONG64_GOT_HI and R_LOONG64_GOT_LO resolves a GOT-relative instruction sequence,
	// usually an pcalau12i followed by another ld or addi instruction.
	R_LOONG64_GOT_HI
	R_LOONG64_GOT_LO

	// R_JMPLOONG64 resolves to non-PC-relative target address of a JMP instruction,
	// by encoding the address into the instruction.
	R_JMPLOONG64

	// R_ADDRMIPSU (only used on mips/mips64) resolves to the sign-adjusted "upper" 16
	// bits (bit 16-31) of an external address, by encoding it into the instruction.
	R_ADDRMIPSU
	// R_ADDRMIPSTLS (only used on mips64) resolves to the low 16 bits of a TLS
	// address (offset from thread pointer), by encoding it into the instruction.
	R_ADDRMIPSTLS

	// R_ADDRCUOFF resolves to a pointer-sized offset from the start of the
	// symbol's DWARF compile unit.
	R_ADDRCUOFF

	// R_WASMIMPORT resolves to the index of the WebAssembly function import.
	R_WASMIMPORT

	// R_XCOFFREF (only used on aix/ppc64) prevents garbage collection by ld
	// of a symbol. This isn't a real relocation, it can be placed in anywhere
	// in a symbol and target any symbols.
	R_XCOFFREF

	// R_PEIMAGEOFF resolves to a 32-bit offset from the start address of where
	// the executable file is mapped in memory.
	R_PEIMAGEOFF

	// R_INITORDER specifies an ordering edge between two inittask records.
	// (From one p..inittask record to another one.)
	// This relocation does not apply any changes to the actual data, it is
	// just used in the linker to order the inittask records appropriately.
	R_INITORDER

	// R_WEAK marks the relocation as a weak reference.
	// A weak relocation does not make the symbol it refers to reachable,
	// and is only honored by the linker if the symbol is in some other way
	// reachable.
	R_WEAK = -1 << 15

	R_WEAKADDR    = R_WEAK | R_ADDR
	R_WEAKADDROFF = R_WEAK | R_ADDROFF
)

// IsDirectCall reports whether r is a relocation for a direct call.
// A direct call is a CALL instruction that takes the target address
// as an immediate. The address is embedded into the instruction(s), possibly
// with limited width. An indirect call is a CALL instruction that takes
// the target address in register or memory.
func (r RelocType) IsDirectCall() bool {
	switch r {
	case R_CALL, R_CALLARM, R_CALLARM64, R_CALLLOONG64, R_CALLMIPS, R_CALLPOWER,
		R_RISCV_CALL, R_RISCV_JAL, R_RISCV_JAL_TRAMP:
		return true
	}
	return false
}

// IsDirectJump reports whether r is a relocation for a direct jump.
// A direct jump is a JMP instruction that takes the target address
// as an immediate. The address is embedded into the instruction, possibly
// with limited width. An indirect jump is a JMP instruction that takes
// the target address in register or memory.
func (r RelocType) IsDirectJump() bool {
	switch r {
	case R_JMPMIPS:
		return true
	case R_JMPLOONG64:
		return true
	}
	return false
}

// IsDirectCallOrJump reports whether r is a relocation for a direct
// call or a direct jump.
func (r RelocType) IsDirectCallOrJump() bool {
	return r.IsDirectCall() || r.IsDirectJump()
}
