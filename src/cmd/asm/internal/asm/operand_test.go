// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asm

import (
	"testing"

	"cmd/asm/internal/arch"
	"cmd/asm/internal/lex"
	"cmd/internal/obj"
)

// A simple in-out test: Do we print what we parse?

func TestAMD64OperandParser(t *testing.T) {
	architecture := arch.Set("amd64")
	ctxt := obj.Linknew(architecture.LinkArch)
	parser := NewParser(ctxt, architecture, nil)
	for _, test := range amd64operandTests {
		parser.start(lex.Tokenize(test.input))
		addr := obj.Addr{}
		parser.operand(&addr)
		result := parser.arch.Dconv(&emptyProg, 0, &addr)
		if result != test.output {
			t.Errorf("fail at %s: got %s; expected %s\n", test.input, result, test.output)
		}
	}
}

type operandTest struct {
	input, output string
}

// Examples collected by scanning all the assembly in the standard repo.

var amd64operandTests = []operandTest{
	{"$(-1.0)", "$(-1)"}, // TODO: Should print as a float.
	{"$(0.0)", "$(0)"},   // TODO: Should print as a float.
	{"$(0x2000000+116)", "$33554548"},
	{"$(0x3F<<7)", "$8064"},
	{"$(112+8)", "$120"},
	{"$(1<<63)", "$-9223372036854775808"},
	{"$-1", "$-1"},
	{"$0", "$0"},
	{"$0-0", "$0"},
	{"$0-16", "$-16"},
	{"$0x000FFFFFFFFFFFFF", "$4503599627370495"},
	{"$0x01", "$1"},
	{"$0x02", "$2"},
	{"$0x04", "$4"},
	{"$0x3FE", "$1022"},
	{"$0x7fffffe00000", "$140737486258176"},
	{"$0xfffffffffffff001", "$-4095"},
	{"$1", "$1"},
	{"$1.0", "$(1)"}, // TODO: should print as float.
	{"$10", "$10"},
	{"$1000", "$1000"},
	{"$1000000", "$1000000"},
	{"$1000000000", "$1000000000"},
	{"$__tsan_func_enter(SB)", "$__tsan_func_enter+0(SB)"},
	{"$main(SB)", "$main+0(SB)"},
	{"$masks<>(SB)", "$masks<>+0(SB)"},
	{"$setg_gcc<>(SB)", "$setg_gcc<>+0(SB)"},
	{"$shifts<>(SB)", "$shifts<>+0(SB)"},
	{"$~(1<<63)", "$9223372036854775807"},
	{"$~0x3F", "$-64"},
	{"$~15", "$-16"},
	{"(((8)&0xf)*4)(SP)", "32(SP)"},
	{"(((8-14)&0xf)*4)(SP)", "40(SP)"},
	{"(AX)", "(AX)"},
	{"(AX)(CX*8)", "(AX)(CX*8)"},
	{"(BP)(CX*4)", "(BP)(CX*4)"},
	{"(BP)(DX*4)", "(BP)(DX*4)"},
	{"(BP)(R8*4)", "(BP)(R8*4)"},
	{"(BX)", "(BX)"},
	{"(DI)", "(DI)"},
	{"(DI)(BX*1)", "(DI)(BX*1)"},
	{"(DX)", "(DX)"},
	{"(R9)", "(R9)"},
	{"(R9)(BX*8)", "(R9)(BX*8)"},
	{"(SI)", "(SI)"},
	{"(SI)(BX*1)", "(SI)(BX*1)"},
	{"(SI)(DX*1)", "(SI)(DX*1)"},
	{"(SP)", "(SP)"},
	{"(6+8)(AX)", "14(AX)"},
	{"(8*4)(BP)", "32(BP)"},
	// {"+3(PC)", "+3(PC)"}, TODO: Need to parse this knowing it's a branch.
	{"-1(DI)(BX*1)", "-1(DI)(BX*1)"},
	{"-64(SI)(BX*1)", "-64(SI)(BX*1)"},
	{"-96(SI)(BX*1)", "-96(SI)(BX*1)"},
	{"AL", "AL"},
	{"AX", "AX"},
	// {"AX:DX", "AX:DX"}, TODO: prints as AX although -S output is correct.
	{"BP", "BP"},
	{"BX", "BX"},
	{"CX", "CX"},
	{"DI", "DI"},
	{"DX", "DX"},
	{"R10", "R10"},
	{"R11", "R11"},
	{"R12", "R12"},
	{"R13", "R13"},
	{"R14", "R14"},
	{"R15", "R15"},
	{"R8", "R8"},
	{"R9", "R9"},
	{"SI", "SI"},
	{"SP", "SP"},
	{"X0", "X0"},
	{"X1", "X1"},
	{"X10", "X10"},
	{"X11", "X11"},
	{"X12", "X12"},
	{"X13", "X13"},
	{"X14", "X14"},
	{"X15", "X15"},
	{"X2", "X2"},
	{"X3", "X3"},
	{"X4", "X4"},
	{"X5", "X5"},
	{"X6", "X6"},
	{"X7", "X7"},
	{"X8", "X8"},
	{"X9", "X9"},
	{"_expand_key_128<>(SB)", "_expand_key_128<>+0(SB)"},
	{"_seek<>(SB)", "_seek<>+0(SB)"},
	{"a2+16(FP)", "a2+16(FP)"},
	{"addr2+24(FP)", "addr2+24(FP)"},
	{"asmcgocall<>(SB)", "asmcgocall<>+0(SB)"},
	{"b+24(FP)", "b+24(FP)"},
	{"b_len+32(FP)", "b_len+32(FP)"},
	{"racecall<>(SB)", "racecall<>+0(SB)"},
	{"rcv_name+20(FP)", "rcv_name+20(FP)"},
	{"retoffset+28(FP)", "retoffset+28(FP)"},
	{"runtime·_GetStdHandle(SB)", "runtime._GetStdHandle+0(SB)"},
	{"sync\u2215atomic·AddInt64(SB)", "sync/atomic.AddInt64+0(SB)"},
	{"timeout+20(FP)", "timeout+20(FP)"},
	{"ts+16(FP)", "ts+16(FP)"},
	{"x+24(FP)", "x+24(FP)"},
	{"x·y(SB)", "x.y+0(SB)"},
	{"x·y(SP)", "x.y+0(SP)"},
	{"x·y+8(SB)", "x.y+8(SB)"},
	{"x·y+8(SP)", "x.y+8(SP)"},
	{"y+56(FP)", "y+56(FP)"},
	{"·AddUint32(SB", "\"\".AddUint32+0(SB)"},
	{"·callReflect(SB)", "\"\".callReflect+0(SB)"},
}
