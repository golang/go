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
	AGet = obj.ABaseWasm + obj.A_ARCHSPECIFIC + iota
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

	AMemoryInit
	ADataDrop
	AMemoryCopy
	AMemoryFill
	ATableInit
	AElemDrop
	ATableCopy
	ATableGrow
	ATableSize
	ATableFill

	// WASM Vector instructions 0xFD 0x00 and following
	AV128Load
	AV128Load8x8S
	AV128Load8x8U
	AV128Load16x4S
	AV128Load16x4U
	AV128Load32x2S
	AV128Load32x2U
	AV128Load8Splat
	AV128Load16Splat
	AV128Load32Splat
	AV128Load64Splat
	AV128Store
	AV128Const
	AI8x16Shuffle16
	AI8x16Swizzle
	AI8x16Splat
	AI16x8Splat
	AI32x4Splat
	AI64x2Splat
	AF32x4Splat
	AF64x2Splat
	AI8x16ExtractLaneS
	AI8x16ExtractLaneU
	AI8x16ReplaceLane
	AI16x8ExtractLaneS
	AI16x8ExtractLaneU
	AI16x8ReplaceLane
	AI32x4ExtractLane
	AI32x4ReplaceLane
	AI64x2ExtractLane
	AI64x2ReplaceLane
	AF32x4ExtractLane
	AF32x4ReplaceLane
	AF64x2ExtractLane
	AF64x2ReplaceLane
	AI8x16Eq
	AI8x16Ne
	AI8x16LtS
	AI8x16LtU
	AI8x16GtS
	AI8x16GtU
	AI8x16LeS
	AI8x16LeU
	AI8x16GeS
	AI8x16GeU
	AI16x8Eq
	AI16x8Ne
	AI16x8LtS
	AI16x8LtU
	AI16x8GtS
	AI16x8GtU
	AI16x8LeS
	AI16x8LeU
	AI16x8GeS
	AI16x8GeU
	AI32x4Eq
	AI32x4Ne
	AI32x4LtS
	AI32x4LtU
	AI32x4GtS
	AI32x4GtU
	AI32x4LeS
	AI32x4LeU
	AI32x4GeS
	AI32x4GeU
	AF32x4Eq
	AF32x4Ne
	AF32x4Lt
	AF32x4Gt
	AF32x4Le
	AF32x4Ge
	AF64x2Eq
	AF64x2Ne
	AF64x2Lt
	AF64x2Gt
	AF64x2Le
	AF64x2Ge
	AV128Not
	AV128And
	AV128Andnot
	AV128Or
	AV128Xor
	AV128Bitselect
	AV128AnyTrue
	AV128Load8Lane
	AV128Load16Lane
	AV128Load32Lane
	AV128Load64Lane
	AV128Store8Lane
	AV128Store16Lane
	AV128Store32Lane
	AV128Store64Lane
	AV128Load32Zero
	AV128Load64Zero
	AF32x4DemoteF64x2Zero
	AF64x2PromoteLowF32x4
	AI8x16Abs
	AI8x16Neg
	AI8x16Popcnt
	AI8x16AllTrue
	AI8x16Bitmask
	AI8x16NarrowI16x8S
	AI8x16NarrowI16x8U
	AF32x4Ceil
	AF32x4Floor
	AF32x4Trunc
	AF32x4Nearest
	AI8x16Shl
	AI8x16ShrS
	AI8x16ShrU
	AI8x16Add
	AI8x16AddSatS
	AI8x16AddSatU
	AI8x16Sub
	AI8x16SubSatS
	AI8x16SubSatU
	AF64x2Ceil
	AF64x2Floor
	AI8x16MinS
	AI8x16MinU
	AI8x16MaxS
	AI8x16MaxU
	AF64x2Trunc
	AI8x16AvgrU
	AI16x8ExtaddPairwiseI8x16S
	AI16x8ExtaddPairwiseI8x16U
	AI32x4ExtaddPairwiseI16x8S
	AI32x4ExtaddPairwiseI16x8U

	// WASM Vector instructions 0xFD 0x80 and following but followed by a 0x01
	// e.g. 0xFD 0x80 0x01, 0xFD  0x81  0x01, etc.
	AI16x8Abs
	AI16x8Neg
	AI16x8Q15MulrSatS
	AI16x8AllTrue
	AI16x8Bitmask
	AI16x8NarrowI32x4S
	AI16x8NarrowI32x4U
	AI16x8ExtendLowI8x16S
	AI16x8ExtendHighI8x16S
	AI16x8ExtendLowI8x16U
	AI16x8ExtendHighI8x16U
	AI16x8Shl
	AI16x8ShrS
	AI16x8ShrU
	AI16x8Add
	AI16x8AddSatS
	AI16x8AddSatU
	AI16x8Sub
	AI16x8SubSatS
	AI16x8SubSatU
	AF64x2Nearest
	AI16x8Mul
	AI16x8MinS
	AI16x8MinU
	AI16x8MaxS
	AI16x8MaxU
	AReservedFD9A01 // 0xFD 0x9A 0x01
	AI16x8AvgrU
	AI16x8ExtmulLowI8x16S
	AI16x8ExtmulHighI8x16S
	AI16x8ExtmulLowI8x16U
	AI16x8ExtmulHighI8x16U
	AI32x4Abs
	AI32x4Neg
	AReservedFDA201
	AI32x4AllTrue
	AI32x4Bitmask
	AReservedFDA501
	AReservedFDA601
	AI32x4ExtendLowI16x8S
	AI32x4ExtendHighI16x8S
	AI32x4ExtendLowI16x8U
	AI32x4ExtendHighI16x8U
	AI32x4Shl
	AI32x4ShrS
	AI32x4ShrU
	AI32x4Add
	AReservedFDAF01
	AReservedFDB001
	AI32x4Sub
	AReservedFDB201
	AReservedFDB301
	AReservedFDB401
	AI32x4Mul
	AI32x4MinS
	AI32x4MinU
	AI32x4MaxS
	AI32x4MaxU
	AI32x4DotI16x8S
	AReservedFDBB01
	AI32x4ExtmulLowI16x8S
	AI32x4ExtmulHighI16x8S
	AI32x4ExtmulLowI16x8U
	AI32x4ExtmulHighI16x8U
	AI64x2Abs
	AI64x2Neg
	AReservedFDC201
	AI64x2AllTrue
	AI64x2Bitmask
	AReservedFDC501
	AReservedFDC601
	AI64x2ExtendLowI32x4S
	AI64x2ExtendHighI32x4S
	AI64x2ExtendLowI32x4U
	AI64x2ExtendHighI32x4U
	AI64x2Shl
	AI64x2ShrS
	AI64x2ShrU
	AI64x2Add
	AReservedFDCF01
	AReservedFDD001
	AI64x2Sub
	AReservedFDD201
	AReservedFDD301
	AReservedFDD401
	AI64x2Mul
	AI64x2Eq
	AI64x2Ne
	AI64x2LtS
	AI64x2GtS
	AI64x2LeS
	AI64x2GeS
	AI64x2ExtmulLowI32x4S
	AI64x2ExtmulHighI32x4S
	AI64x2ExtmulLowI32x4U
	AI64x2ExtmulHighI32x4U
	AF32x4Abs
	AF32x4Neg
	AReservedFDE201
	AF32x4Sqrt
	AF32x4Add
	AF32x4Sub
	AF32x4Mul
	AF32x4Div
	AF32x4Min
	AF32x4Max
	AF32x4Pmin
	AF32x4Pmax
	AF64x2Abs
	AF64x2Neg
	AReservedFDEE01
	AF64x2Sqrt
	AF64x2Add
	AF64x2Sub
	AF64x2Mul
	AF64x2Div
	AF64x2Min
	AF64x2Max
	AF64x2Pmin
	AF64x2Pmax
	AI32x4TruncSatF32x4S
	AI32x4TruncSatF32x4U
	AF32x4ConvertI32x4S
	AF32x4ConvertI32x4U
	AI32x4TruncSatF64x2SZero
	AI32x4TruncSatF64x2UZero
	AF64x2ConvertLowI32x4S
	AF64x2ConvertLowI32x4U

	// WASM Vector instructions 0xFD 0x80 and following but followed by a 0x02
	// e.g. 0xFD 0x80 0x02, 0xFD 0x81 0x02, etc.
	AI8x16RelaxedSwizzle
	AI32x4RelaxedTruncF32x4S
	AI32x4RelaxedTruncF32x4U
	AI32x4RelaxedTruncF64x2S
	AI32x4RelaxedTruncF64x2U
	AF32x4RelaxedMadd
	AF32x4RelaxedNmadd
	AF64x2RelaxedMadd
	AF64x2RelaxedNmadd
	AI8x16RelaxedLaneselect
	AI16x8RelaxedLaneselect
	AI32x4RelaxedLaneselect
	AI64x2RelaxedLaneselect
	AF32x4RelaxedMin
	AF32x4RelaxedMax
	AF64x2RelaxedMin
	AF64x2RelaxedMax
	AI16x8RelaxedQ15MulrS
	AI16x8RelaxedDotI8x16I7x16S
	AI32x4RelaxedDotI8x16I7x16AddS

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

	// v128 locals
	REG_V0
	REG_V1
	REG_V2
	REG_V3
	REG_V4
	REG_V5
	REG_V6
	REG_V7
	REG_V8
	REG_V9
	REG_V10
	REG_V11
	REG_V12
	REG_V13
	REG_V14
	REG_V15

	REG_PC_B // also first parameter, i32

	MAXREG

	MINREG  = REG_SP
	REGSP   = REG_SP
	REGCTXT = REG_CTXT
	REGG    = REG_g
)
