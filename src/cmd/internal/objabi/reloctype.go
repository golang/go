// Derived from Inferno utils/6l/l.h and related files.
// https://bitbucket.org/inferno-os/inferno-os/src/default/utils/6l/l.h
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
	// R_WEAKADDROFF resolves just like R_ADDROFF but is a weak relocation.
	// A weak relocation does not make the symbol it refers to reachable,
	// and is only honored by the linker if the symbol is in some other way
	// reachable.
	R_WEAKADDROFF
	R_SIZE
	R_CALL
	R_CALLARM
	R_CALLARM64
	R_CALLIND
	R_CALLPOWER
	// R_CALLMIPS (only used on mips64) resolves to non-PC-relative target address
	// of a CALL (JAL) instruction, by encoding the address into the instruction.
	R_CALLMIPS
	// R_CALLRISCV marks RISC-V CALLs for stack checking.
	R_CALLRISCV
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
	// R_METHODOFF resolves to a 32-bit offset from the beginning of the section
	// holding the data being relocated to the referenced symbol.
	// It is a variant of R_ADDROFF used when linking from the uncommonType of a
	// *rtype, and may be set to zero by the linker if it determines the method
	// text is unreachable by the linked program.
	R_METHODOFF
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

	// R_ARM64_LDST8 sets a LD/ST immediate value to bits [11:0] of a local address.
	R_ARM64_LDST8

	// R_ARM64_LDST32 sets a LD/ST immediate value to bits [11:2] of a local address.
	R_ARM64_LDST32

	// R_ARM64_LDST64 sets a LD/ST immediate value to bits [11:3] of a local address.
	R_ARM64_LDST64

	// R_ARM64_LDST128 sets a LD/ST immediate value to bits [11:4] of a local address.
	R_ARM64_LDST128

	// PPC64.

	// R_POWER_TLS_LE is used to implement the "local exec" model for tls
	// access. It resolves to the offset of the thread-local symbol from the
	// thread pointer (R13) and inserts this value into the low 16 bits of an
	// instruction word.
	R_POWER_TLS_LE

	// R_POWER_TLS_IE is used to implement the "initial exec" model for tls access. It
	// relocates a D-form, DS-form instruction sequence like R_ADDRPOWER_DS. It
	// inserts to the offset of GOT slot for the thread-local symbol from the TOC (the
	// GOT slot is filled by the dynamic linker with the offset of the thread-local
	// symbol from the thread pointer (R13)).
	R_POWER_TLS_IE

	// R_POWER_TLS marks an X-form instruction such as "MOVD 0(R13)(R31*1), g" as
	// accessing a particular thread-local symbol. It does not affect code generation
	// but is used by the system linker when relaxing "initial exec" model code to
	// "local exec" model code.
	R_POWER_TLS

	// R_ADDRPOWER_DS is similar to R_ADDRPOWER above, but assumes the second
	// instruction is a "DS-form" instruction, which has an immediate field occupying
	// bits [15:2] of the instruction word. Bits [15:2] of the address of the
	// relocated symbol are inserted into this field; it is an error if the last two
	// bits of the address are not 0.
	R_ADDRPOWER_DS

	// R_ADDRPOWER_PCREL relocates a D-form, DS-form instruction sequence like
	// R_ADDRPOWER_DS but inserts the offset of the GOT slot for the referenced symbol
	// from the TOC rather than the symbol's address.
	R_ADDRPOWER_GOT

	// R_ADDRPOWER_PCREL relocates two D-form instructions like R_ADDRPOWER, but
	// inserts the displacement from the place being relocated to the address of the
	// relocated symbol instead of just its address.
	R_ADDRPOWER_PCREL

	// R_ADDRPOWER_TOCREL relocates two D-form instructions like R_ADDRPOWER, but
	// inserts the offset from the TOC to the address of the relocated symbol
	// rather than the symbol's address.
	R_ADDRPOWER_TOCREL

	// R_ADDRPOWER_TOCREL relocates a D-form, DS-form instruction sequence like
	// R_ADDRPOWER_DS but inserts the offset from the TOC to the address of the
	// relocated symbol rather than the symbol's address.
	R_ADDRPOWER_TOCREL_DS

	// RISC-V.

	// R_RISCV_PCREL_ITYPE resolves a 32-bit PC-relative address using an
	// AUIPC + I-type instruction pair.
	R_RISCV_PCREL_ITYPE

	// R_RISCV_PCREL_STYPE resolves a 32-bit PC-relative address using an
	// AUIPC + S-type instruction pair.
	R_RISCV_PCREL_STYPE

	// R_PCRELDBL relocates s390x 2-byte aligned PC-relative addresses.
	// TODO(mundaym): remove once variants can be serialized - see issue 14218.
	R_PCRELDBL

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
)

// IsDirectCall reports whether r is a relocation for a direct call.
// A direct call is a CALL instruction that takes the target address
// as an immediate. The address is embedded into the instruction, possibly
// with limited width. An indirect call is a CALL instruction that takes
// the target address in register or memory.
func (r RelocType) IsDirectCall() bool {
	switch r {
	case R_CALL, R_CALLARM, R_CALLARM64, R_CALLMIPS, R_CALLPOWER, R_CALLRISCV:
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
	}
	return false
}

// IsDirectCallOrJump reports whether r is a relocation for a direct
// call or a direct jump.
func (r RelocType) IsDirectCallOrJump() bool {
	return r.IsDirectCall() || r.IsDirectJump()
}
