// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package wasm

import "cmd/internal/obj"

//go:generate go run ../stringer.go -i $GOFILE -o anames.go -p wasm

const (
	/* mark flags */
	DONE          = 1 << iota
	PRESERVEFLAGS // not allowed to clobber flags
)

/*
 *	wasm
 */
const (
	ACallImport = obj.ABaseWasm + obj.A_ARCHSPECIFIC + iota
	AGet
	ASet
	ATee
	ANot // alias for I32Eqz

	// The following are low-level WebAssembly instructions.
	// Their order matters, since it matches the opcode encoding.
	// Gaps in the encoding are indicated by comments.

	AUnreachable // opcode 0x00
	ANop
	ABlock
	ALoop
	AIf
	AElse

	AEnd // opcode 0x0B
	ABr
	ABrIf
	ABrTable
	// ACall and AReturn are WebAssembly instructions. obj.ACALL and obj.ARET are higher level instructions
	// with Go semantics, e.g. they manipulate the Go stack on the linear memory.
	AReturn
	ACall
	ACallIndirect

	ADrop // opcode 0x1A
	ASelect

	ALocalGet // opcode 0x20
	ALocalSet
	ALocalTee
	AGlobalGet
	AGlobalSet

	AI32Load // opcode 0x28
	AI64Load
	AF32Load
	AF64Load
	AI32Load8S
	AI32Load8U
	AI32Load16S
	AI32Load16U
	AI64Load8S
	AI64Load8U
	AI64Load16S
	AI64Load16U
	AI64Load32S
	AI64Load32U
	AI32Store
	AI64Store
	AF32Store
	AF64Store
	AI32Store8
	AI32Store16
	AI64Store8
	AI64Store16
	AI64Store32
	ACurrentMemory
	AGrowMemory

	AI32Const
	AI64Const
	AF32Const
	AF64Const

	AI32Eqz
	AI32Eq
	AI32Ne
	AI32LtS
	AI32LtU
	AI32GtS
	AI32GtU
	AI32LeS
	AI32LeU
	AI32GeS
	AI32GeU

	AI64Eqz
	AI64Eq
	AI64Ne
	AI64LtS
	AI64LtU
	AI64GtS
	AI64GtU
	AI64LeS
	AI64LeU
	AI64GeS
	AI64GeU

	AF32Eq
	AF32Ne
	AF32Lt
	AF32Gt
	AF32Le
	AF32Ge

	AF64Eq
	AF64Ne
	AF64Lt
	AF64Gt
	AF64Le
	AF64Ge

	AI32Clz
	AI32Ctz
	AI32Popcnt
	AI32Add
	AI32Sub
	AI32Mul
	AI32DivS
	AI32DivU
	AI32RemS
	AI32RemU
	AI32And
	AI32Or
	AI32Xor
	AI32Shl
	AI32ShrS
	AI32ShrU
	AI32Rotl
	AI32Rotr

	AI64Clz
	AI64Ctz
	AI64Popcnt
	AI64Add
	AI64Sub
	AI64Mul
	AI64DivS
	AI64DivU
	AI64RemS
	AI64RemU
	AI64And
	AI64Or
	AI64Xor
	AI64Shl
	AI64ShrS
	AI64ShrU
	AI64Rotl
	AI64Rotr

	AF32Abs
	AF32Neg
	AF32Ceil
	AF32Floor
	AF32Trunc
	AF32Nearest
	AF32Sqrt
	AF32Add
	AF32Sub
	AF32Mul
	AF32Div
	AF32Min
	AF32Max
	AF32Copysign

	AF64Abs
	AF64Neg
	AF64Ceil
	AF64Floor
	AF64Trunc
	AF64Nearest
	AF64Sqrt
	AF64Add
	AF64Sub
	AF64Mul
	AF64Div
	AF64Min
	AF64Max
	AF64Copysign

	AI32WrapI64
	AI32TruncF32S
	AI32TruncF32U
	AI32TruncF64S
	AI32TruncF64U
	AI64ExtendI32S
	AI64ExtendI32U
	AI64TruncF32S
	AI64TruncF32U
	AI64TruncF64S
	AI64TruncF64U
	AF32ConvertI32S
	AF32ConvertI32U
	AF32ConvertI64S
	AF32ConvertI64U
	AF32DemoteF64
	AF64ConvertI32S
	AF64ConvertI32U
	AF64ConvertI64S
	AF64ConvertI64U
	AF64PromoteF32
	AI32ReinterpretF32
	AI64ReinterpretF64
	AF32ReinterpretI32
	AF64ReinterpretI64
	AI32Extend8S
	AI32Extend16S
	AI64Extend8S
	AI64Extend16S
	AI64Extend32S

	AI32TruncSatF32S // opcode 0xFC 0x00
	AI32TruncSatF32U
	AI32TruncSatF64S
	AI32TruncSatF64U
	AI64TruncSatF32S
	AI64TruncSatF32U
	AI64TruncSatF64S
	AI64TruncSatF64U

	ALast // Sentinel: End of low-level WebAssembly instructions.

	ARESUMEPOINT
	// ACALLNORESUME is a call which is not followed by a resume point.
	// It is allowed inside of WebAssembly blocks, whereas obj.ACALL is not.
	// However, it is not allowed to switch goroutines while inside of an ACALLNORESUME call.
	ACALLNORESUME

	ARETUNWIND

	AMOVB
	AMOVH
	AMOVW
	AMOVD

	AWORD
	ALAST
)

const (
	REG_NONE = 0
)

const (
	// globals
	REG_SP = obj.RBaseWasm + iota // SP is currently 32-bit, until 64-bit memory operations are available
	REG_CTXT
	REG_g
	// RET* are used by runtime.return0 and runtime.reflectcall. These functions pass return values in registers.
	REG_RET0
	REG_RET1
	REG_RET2
	REG_RET3
	REG_PAUSE

	// i32 locals
	REG_R0
	REG_R1
	REG_R2
	REG_R3
	REG_R4
	REG_R5
	REG_R6
	REG_R7
	REG_R8
	REG_R9
	REG_R10
	REG_R11
	REG_R12
	REG_R13
	REG_R14
	REG_R15

	// f32 locals
	REG_F0
	REG_F1
	REG_F2
	REG_F3
	REG_F4
	REG_F5
	REG_F6
	REG_F7
	REG_F8
	REG_F9
	REG_F10
	REG_F11
	REG_F12
	REG_F13
	REG_F14
	REG_F15

	// f64 locals
	REG_F16
	REG_F17
	REG_F18
	REG_F19
	REG_F20
	REG_F21
	REG_F22
	REG_F23
	REG_F24
	REG_F25
	REG_F26
	REG_F27
	REG_F28
	REG_F29
	REG_F30
	REG_F31

	REG_PC_B // also first parameter, i32

	MAXREG

	MINREG  = REG_SP
	REGSP   = REG_SP
	REGCTXT = REG_CTXT
	REGG    = REG_g
)
