// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ignore

package main

import "strings"

var regNamesWasm = []string{
	"R0",
	"R1",
	"R2",
	"R3",
	"R4",
	"R5",
	"R6",
	"R7",
	"R8",
	"R9",
	"R10",
	"R11",
	"R12",
	"R13",
	"R14",
	"R15",

	"F0",
	"F1",
	"F2",
	"F3",
	"F4",
	"F5",
	"F6",
	"F7",
	"F8",
	"F9",
	"F10",
	"F11",
	"F12",
	"F13",
	"F14",
	"F15",

	"SP",
	"g",

	// pseudo-registers
	"SB",
}

func init() {
	// Make map from reg names to reg integers.
	if len(regNamesWasm) > 64 {
		panic("too many registers")
	}
	num := map[string]int{}
	for i, name := range regNamesWasm {
		num[name] = i
	}
	buildReg := func(s string) regMask {
		m := regMask(0)
		for _, r := range strings.Split(s, " ") {
			if n, ok := num[r]; ok {
				m |= regMask(1) << uint(n)
				continue
			}
			panic("register " + r + " not found")
		}
		return m
	}

	var (
		gp     = buildReg("R0 R1 R2 R3 R4 R5 R6 R7 R8 R9 R10 R11 R12 R13 R14 R15")
		fp     = buildReg("F0 F1 F2 F3 F4 F5 F6 F7 F8 F9 F10 F11 F12 F13 F14 F15")
		gpsp   = gp | buildReg("SP")
		gpspsb = gpsp | buildReg("SB")
		// The "registers", which are actually local variables, can get clobbered
		// if we're switching goroutines, because it unwinds the WebAssembly stack.
		callerSave = gp | fp | buildReg("g")
	)

	// Common regInfo
	var (
		gp01    = regInfo{inputs: nil, outputs: []regMask{gp}}
		gp11    = regInfo{inputs: []regMask{gpsp}, outputs: []regMask{gp}}
		gp21    = regInfo{inputs: []regMask{gpsp, gpsp}, outputs: []regMask{gp}}
		gp31    = regInfo{inputs: []regMask{gpsp, gpsp, gpsp}, outputs: []regMask{gp}}
		fp01    = regInfo{inputs: nil, outputs: []regMask{fp}}
		fp11    = regInfo{inputs: []regMask{fp}, outputs: []regMask{fp}}
		fp21    = regInfo{inputs: []regMask{fp, fp}, outputs: []regMask{fp}}
		fp21gp  = regInfo{inputs: []regMask{fp, fp}, outputs: []regMask{gp}}
		gpload  = regInfo{inputs: []regMask{gpspsb, 0}, outputs: []regMask{gp}}
		gpstore = regInfo{inputs: []regMask{gpspsb, gpsp, 0}}
		fpload  = regInfo{inputs: []regMask{gpspsb, 0}, outputs: []regMask{fp}}
		fpstore = regInfo{inputs: []regMask{gpspsb, fp, 0}}
		// fpstoreconst = regInfo{inputs: []regMask{fp, 0}}
	)

	var WasmOps = []opData{
		{name: "LoweredStaticCall", argLength: 1, reg: regInfo{clobbers: callerSave}, aux: "SymOff", call: true, symEffect: "None"},            // call static function aux.(*obj.LSym). arg0=mem, auxint=argsize, returns mem
		{name: "LoweredClosureCall", argLength: 3, reg: regInfo{inputs: []regMask{gp, gp, 0}, clobbers: callerSave}, aux: "Int64", call: true}, // call function via closure. arg0=codeptr, arg1=closure, arg2=mem, auxint=argsize, returns mem
		{name: "LoweredInterCall", argLength: 2, reg: regInfo{inputs: []regMask{gp}, clobbers: callerSave}, aux: "Int64", call: true},          // call fn by pointer. arg0=codeptr, arg1=mem, auxint=argsize, returns mem

		{name: "LoweredAddr", argLength: 1, reg: gp11, aux: "SymOff", rematerializeable: true, symEffect: "Addr"}, // returns base+aux+auxint, arg0=base
		{name: "LoweredMove", argLength: 3, reg: regInfo{inputs: []regMask{gp, gp}}, aux: "Int64"},                // large move. arg0=dst, arg1=src, arg2=mem, auxint=len/8, returns mem
		{name: "LoweredZero", argLength: 2, reg: regInfo{inputs: []regMask{gp}}, aux: "Int64"},                    // large zeroing. arg0=start, arg1=mem, auxint=len/8, returns mem

		{name: "LoweredGetClosurePtr", reg: gp01},                                                                          // returns wasm.REG_CTXT, the closure pointer
		{name: "LoweredGetCallerPC", reg: gp01, rematerializeable: true},                                                   // returns the PC of the caller of the current function
		{name: "LoweredGetCallerSP", reg: gp01, rematerializeable: true},                                                   // returns the SP of the caller of the current function
		{name: "LoweredNilCheck", argLength: 2, reg: regInfo{inputs: []regMask{gp}}, nilCheck: true, faultOnNilArg0: true}, // panic if arg0 is nil. arg1=mem
		{name: "LoweredWB", argLength: 3, reg: regInfo{inputs: []regMask{gp, gp}}, aux: "Sym", symEffect: "None"},          // invokes runtime.gcWriteBarrier. arg0=destptr, arg1=srcptr, arg2=mem, aux=runtime.gcWriteBarrier
		{name: "LoweredRound32F", argLength: 1, reg: fp11, typ: "Float32"},                                                 // rounds arg0 to 32-bit float precision. arg0=value

		// LoweredConvert converts between pointers and integers.
		// We have a special op for this so as to not confuse GC
		// (particularly stack maps). It takes a memory arg so it
		// gets correctly ordered with respect to GC safepoints.
		// arg0=ptr/int arg1=mem, output=int/ptr
		//
		// TODO(neelance): LoweredConvert should not be necessary any more, since OpConvert does not need to be lowered any more (CL 108496).
		{name: "LoweredConvert", argLength: 2, reg: regInfo{inputs: []regMask{gp}, outputs: []regMask{gp}}},

		// The following are native WebAssembly instructions, see https://webassembly.github.io/spec/core/syntax/instructions.html

		{name: "Select", asm: "Select", argLength: 3, reg: gp31}, // returns arg0 if arg2 != 0, otherwise returns arg1

		{name: "I64Load8U", asm: "I64Load8U", argLength: 2, reg: gpload, aux: "Int64", typ: "UInt8"},    // read unsigned 8-bit integer from address arg0+aux, arg1=mem
		{name: "I64Load8S", asm: "I64Load8S", argLength: 2, reg: gpload, aux: "Int64", typ: "Int8"},     // read signed 8-bit integer from address arg0+aux, arg1=mem
		{name: "I64Load16U", asm: "I64Load16U", argLength: 2, reg: gpload, aux: "Int64", typ: "UInt16"}, // read unsigned 16-bit integer from address arg0+aux, arg1=mem
		{name: "I64Load16S", asm: "I64Load16S", argLength: 2, reg: gpload, aux: "Int64", typ: "Int16"},  // read signed 16-bit integer from address arg0+aux, arg1=mem
		{name: "I64Load32U", asm: "I64Load32U", argLength: 2, reg: gpload, aux: "Int64", typ: "UInt32"}, // read unsigned 32-bit integer from address arg0+aux, arg1=mem
		{name: "I64Load32S", asm: "I64Load32S", argLength: 2, reg: gpload, aux: "Int64", typ: "Int32"},  // read signed 32-bit integer from address arg0+aux, arg1=mem
		{name: "I64Load", asm: "I64Load", argLength: 2, reg: gpload, aux: "Int64", typ: "UInt64"},       // read 64-bit integer from address arg0+aux, arg1=mem
		{name: "I64Store8", asm: "I64Store8", argLength: 3, reg: gpstore, aux: "Int64", typ: "Mem"},     // store 8-bit integer arg1 at address arg0+aux, arg2=mem, returns mem
		{name: "I64Store16", asm: "I64Store16", argLength: 3, reg: gpstore, aux: "Int64", typ: "Mem"},   // store 16-bit integer arg1 at address arg0+aux, arg2=mem, returns mem
		{name: "I64Store32", asm: "I64Store32", argLength: 3, reg: gpstore, aux: "Int64", typ: "Mem"},   // store 32-bit integer arg1 at address arg0+aux, arg2=mem, returns mem
		{name: "I64Store", asm: "I64Store", argLength: 3, reg: gpstore, aux: "Int64", typ: "Mem"},       // store 64-bit integer arg1 at address arg0+aux, arg2=mem, returns mem

		{name: "F32Load", asm: "F32Load", argLength: 2, reg: fpload, aux: "Int64", typ: "Float64"}, // read 32-bit float from address arg0+aux, arg1=mem
		{name: "F64Load", asm: "F64Load", argLength: 2, reg: fpload, aux: "Int64", typ: "Float64"}, // read 64-bit float from address arg0+aux, arg1=mem
		{name: "F32Store", asm: "F32Store", argLength: 3, reg: fpstore, aux: "Int64", typ: "Mem"},  // store 32-bit float arg1 at address arg0+aux, arg2=mem, returns mem
		{name: "F64Store", asm: "F64Store", argLength: 3, reg: fpstore, aux: "Int64", typ: "Mem"},  // store 64-bit float arg1 at address arg0+aux, arg2=mem, returns mem

		{name: "I64Const", reg: gp01, aux: "Int64", rematerializeable: true, typ: "Int64"},     // returns the constant integer aux
		{name: "F64Const", reg: fp01, aux: "Float64", rematerializeable: true, typ: "Float64"}, // returns the constant float aux

		{name: "I64Eqz", asm: "I64Eqz", argLength: 1, reg: gp11, typ: "Bool"}, // arg0 == 0
		{name: "I64Eq", asm: "I64Eq", argLength: 2, reg: gp21, typ: "Bool"},   // arg0 == arg1
		{name: "I64Ne", asm: "I64Ne", argLength: 2, reg: gp21, typ: "Bool"},   // arg0 != arg1
		{name: "I64LtS", asm: "I64LtS", argLength: 2, reg: gp21, typ: "Bool"}, // arg0 < arg1 (signed)
		{name: "I64LtU", asm: "I64LtU", argLength: 2, reg: gp21, typ: "Bool"}, // arg0 < arg1 (unsigned)
		{name: "I64GtS", asm: "I64GtS", argLength: 2, reg: gp21, typ: "Bool"}, // arg0 > arg1 (signed)
		{name: "I64GtU", asm: "I64GtU", argLength: 2, reg: gp21, typ: "Bool"}, // arg0 > arg1 (unsigned)
		{name: "I64LeS", asm: "I64LeS", argLength: 2, reg: gp21, typ: "Bool"}, // arg0 <= arg1 (signed)
		{name: "I64LeU", asm: "I64LeU", argLength: 2, reg: gp21, typ: "Bool"}, // arg0 <= arg1 (unsigned)
		{name: "I64GeS", asm: "I64GeS", argLength: 2, reg: gp21, typ: "Bool"}, // arg0 >= arg1 (signed)
		{name: "I64GeU", asm: "I64GeU", argLength: 2, reg: gp21, typ: "Bool"}, // arg0 >= arg1 (unsigned)

		{name: "F64Eq", asm: "F64Eq", argLength: 2, reg: fp21gp, typ: "Bool"}, // arg0 == arg1
		{name: "F64Ne", asm: "F64Ne", argLength: 2, reg: fp21gp, typ: "Bool"}, // arg0 != arg1
		{name: "F64Lt", asm: "F64Lt", argLength: 2, reg: fp21gp, typ: "Bool"}, // arg0 < arg1
		{name: "F64Gt", asm: "F64Gt", argLength: 2, reg: fp21gp, typ: "Bool"}, // arg0 > arg1
		{name: "F64Le", asm: "F64Le", argLength: 2, reg: fp21gp, typ: "Bool"}, // arg0 <= arg1
		{name: "F64Ge", asm: "F64Ge", argLength: 2, reg: fp21gp, typ: "Bool"}, // arg0 >= arg1

		{name: "I64Add", asm: "I64Add", argLength: 2, reg: gp21, typ: "Int64"},                    // arg0 + arg1
		{name: "I64AddConst", asm: "I64Add", argLength: 1, reg: gp11, aux: "Int64", typ: "Int64"}, // arg0 + aux
		{name: "I64Sub", asm: "I64Sub", argLength: 2, reg: gp21, typ: "Int64"},                    // arg0 - arg1
		{name: "I64Mul", asm: "I64Mul", argLength: 2, reg: gp21, typ: "Int64"},                    // arg0 * arg1
		{name: "I64DivS", asm: "I64DivS", argLength: 2, reg: gp21, typ: "Int64"},                  // arg0 / arg1 (signed)
		{name: "I64DivU", asm: "I64DivU", argLength: 2, reg: gp21, typ: "Int64"},                  // arg0 / arg1 (unsigned)
		{name: "I64RemS", asm: "I64RemS", argLength: 2, reg: gp21, typ: "Int64"},                  // arg0 % arg1 (signed)
		{name: "I64RemU", asm: "I64RemU", argLength: 2, reg: gp21, typ: "Int64"},                  // arg0 % arg1 (unsigned)
		{name: "I64And", asm: "I64And", argLength: 2, reg: gp21, typ: "Int64"},                    // arg0 & arg1
		{name: "I64Or", asm: "I64Or", argLength: 2, reg: gp21, typ: "Int64"},                      // arg0 | arg1
		{name: "I64Xor", asm: "I64Xor", argLength: 2, reg: gp21, typ: "Int64"},                    // arg0 ^ arg1
		{name: "I64Shl", asm: "I64Shl", argLength: 2, reg: gp21, typ: "Int64"},                    // arg0 << (arg1 % 64)
		{name: "I64ShrS", asm: "I64ShrS", argLength: 2, reg: gp21, typ: "Int64"},                  // arg0 >> (arg1 % 64) (signed)
		{name: "I64ShrU", asm: "I64ShrU", argLength: 2, reg: gp21, typ: "Int64"},                  // arg0 >> (arg1 % 64) (unsigned)

		{name: "F64Neg", asm: "F64Neg", argLength: 1, reg: fp11, typ: "Float64"}, // -arg0
		{name: "F64Add", asm: "F64Add", argLength: 2, reg: fp21, typ: "Float64"}, // arg0 + arg1
		{name: "F64Sub", asm: "F64Sub", argLength: 2, reg: fp21, typ: "Float64"}, // arg0 - arg1
		{name: "F64Mul", asm: "F64Mul", argLength: 2, reg: fp21, typ: "Float64"}, // arg0 * arg1
		{name: "F64Div", asm: "F64Div", argLength: 2, reg: fp21, typ: "Float64"}, // arg0 / arg1

		{name: "I64TruncSF64", asm: "I64TruncSF64", argLength: 1, reg: regInfo{inputs: []regMask{fp}, outputs: []regMask{gp}}, typ: "Int64"},       // truncates the float arg0 to a signed integer
		{name: "I64TruncUF64", asm: "I64TruncUF64", argLength: 1, reg: regInfo{inputs: []regMask{fp}, outputs: []regMask{gp}}, typ: "Int64"},       // truncates the float arg0 to an unsigned integer
		{name: "F64ConvertSI64", asm: "F64ConvertSI64", argLength: 1, reg: regInfo{inputs: []regMask{gp}, outputs: []regMask{fp}}, typ: "Float64"}, // converts the signed integer arg0 to a float
		{name: "F64ConvertUI64", asm: "F64ConvertUI64", argLength: 1, reg: regInfo{inputs: []regMask{gp}, outputs: []regMask{fp}}, typ: "Float64"}, // converts the unsigned integer arg0 to a float
	}

	archs = append(archs, arch{
		name:            "Wasm",
		pkg:             "cmd/internal/obj/wasm",
		genfile:         "",
		ops:             WasmOps,
		blocks:          nil,
		regnames:        regNamesWasm,
		gpregmask:       gp,
		fpregmask:       fp,
		framepointerreg: -1, // not used
		linkreg:         -1, // not used
	})
}
