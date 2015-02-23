// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package asm

import (
	"os"
	"testing"

	"cmd/asm/internal/arch"
	"cmd/asm/internal/lex"
	"cmd/internal/obj"
)

// A simple in-out test: Do we print what we parse?

func setArch(goarch string) (*arch.Arch, *obj.Link) {
	os.Setenv("GOOS", "linux") // obj can handle this OS for all architectures.
	architecture := arch.Set(goarch)
	if architecture == nil {
		panic("asm: unrecognized architecture " + goarch)
	}
	return architecture, obj.Linknew(architecture.LinkArch)
}

func newParser(goarch string) *Parser {
	architecture, ctxt := setArch(goarch)
	return NewParser(ctxt, architecture, nil)
}

func testOperandParser(t *testing.T, parser *Parser, tests []operandTest) {
	for _, test := range tests {
		parser.start(lex.Tokenize(test.input))
		addr := obj.Addr{}
		parser.operand(&addr)
		result := parser.arch.Dconv(&emptyProg, 0, &addr)
		if result != test.output {
			t.Errorf("fail at %s: got %s; expected %s\n", test.input, result, test.output)
		}
	}
}

func testX86RegisterPair(t *testing.T, parser *Parser) {
	// Special case for AX:DX, which is really two operands so isn't printed correcctly
	// by Aconv, but is OK by the -S output.
	parser.start(lex.Tokenize("AX:BX)"))
	addr := obj.Addr{}
	parser.operand(&addr)
	want := obj.Addr{
		Type:  obj.TYPE_REG,
		Reg:   parser.arch.Register["AX"],
		Class: int8(parser.arch.Register["BX"]), // TODO: clean up how this is encoded in parse.go
	}
	if want != addr {
		t.Errorf("AX:DX: expected %+v got %+v", want, addr)
	}
}

func TestAMD64OperandParser(t *testing.T) {
	parser := newParser("amd64")
	testOperandParser(t, parser, amd64OperandTests)
	testX86RegisterPair(t, parser)
}

func Test386OperandParser(t *testing.T) {
	parser := newParser("386")
	testOperandParser(t, parser, x86OperandTests)
	testX86RegisterPair(t, parser)
}

func TestARMOperandParser(t *testing.T) {
	parser := newParser("arm")
	testOperandParser(t, parser, armOperandTests)
}

func TestPPC64OperandParser(t *testing.T) {
	parser := newParser("ppc64")
	testOperandParser(t, parser, ppc64OperandTests)
	// Special encoding for (R1+R2).
	parser.start(lex.Tokenize("(R1+R2)"))
	addr := obj.Addr{}
	parser.operand(&addr)
	want := obj.Addr{
		Type:  obj.TYPE_MEM,
		Reg:   parser.arch.Register["R1"],
		Scale: int8(parser.arch.Register["R2"]), // TODO: clean up how this is encoded in parse.go
	}
	if want != addr {
		t.Errorf("(R1+R2): expected %+v got %+v", want, addr)
	}
}

type operandTest struct {
	input, output string
}

// Examples collected by scanning all the assembly in the standard repo.

var amd64OperandTests = []operandTest{
	// {"AX:DX", "AX:DX"}, Handled in TestAMD64OperandParser directly.
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
	{"(6+8)(AX)", "14(AX)"},
	{"(8*4)(BP)", "32(BP)"},
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
	{"+3(PC)", "3(PC)"},
	{"-1(DI)(BX*1)", "-1(DI)(BX*1)"},
	{"-3(PC)", "-3(PC)"},
	{"-64(SI)(BX*1)", "-64(SI)(BX*1)"},
	{"-96(SI)(BX*1)", "-96(SI)(BX*1)"},
	{"AL", "AL"},
	{"AX", "AX"},
	{"BP", "BP"},
	{"BX", "BX"},
	{"CX", "CX"},
	{"DI", "DI"},
	{"DX", "DX"},
	{"R10", "R10"},
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

var x86OperandTests = []operandTest{
	{"$(2.928932188134524e-01)", "$(0.29289321881345243)"},
	{"$-1", "$-1"},
	{"$0", "$0"},
	{"$0x00000000", "$0"},
	{"$runtime·badmcall(SB)", "$runtime.badmcall+0(SB)"},
	{"$setg_gcc<>(SB)", "$setg_gcc<>+0(SB)"},
	{"$~15", "$-16"},
	{"(-64*1024+104)(SP)", "-65432(SP)"},
	{"(0*4)(BP)", "(BP)"},
	{"(1*4)(DI)", "4(DI)"},
	{"(4*4)(BP)", "16(BP)"},
	{"(AX)", "(AX)"},
	{"(BP)(CX*4)", "(BP)(CX*4)"},
	{"(BP*8)", "(NONE)(BP*8)"}, // TODO: odd printout.
	{"(BX)", "(BX)"},
	{"(SP)", "(SP)"},
	{"*runtime·_GetStdHandle(SB)", "type=16"}, // TODO: bizarre
	{"-(4+12)(DI)", "-16(DI)"},
	{"-1(DI)(BX*1)", "-1(DI)(BX*1)"},
	{"-96(DI)(BX*1)", "-96(DI)(BX*1)"},
	{"0(AX)", "(AX)"},
	{"0(BP)", "(BP)"},
	{"0(BX)", "(BX)"},
	{"4(AX)", "4(AX)"},
	{"AL", "AL"},
	{"AX", "AX"},
	{"BP", "BP"},
	{"BX", "BX"},
	{"CX", "CX"},
	{"DI", "DI"},
	{"DX", "DX"},
	{"F0", "F0"},
	{"GS", "GS"},
	{"SI", "SI"},
	{"SP", "SP"},
	{"X0", "X0"},
	{"X1", "X1"},
	{"X2", "X2"},
	{"X3", "X3"},
	{"X4", "X4"},
	{"X5", "X5"},
	{"X6", "X6"},
	{"X7", "X7"},
	{"asmcgocall<>(SB)", "asmcgocall<>+0(SB)"},
	{"ax+4(FP)", "ax+4(FP)"},
	{"ptime-12(SP)", "ptime+-12(SP)"},
	{"runtime·_NtWaitForSingleObject(SB)", "runtime._NtWaitForSingleObject+0(SB)"},
	{"s(FP)", "s+0(FP)"},
	{"sec+4(FP)", "sec+4(FP)"},
	{"shifts<>(SB)(CX*8)", "shifts<>+0(SB)(CX*8)"},
	{"x+4(FP)", "x+4(FP)"},
	{"·AddUint32(SB)", "\"\".AddUint32+0(SB)"},
	{"·reflectcall(SB)", "\"\".reflectcall+0(SB)"},
}

var armOperandTests = []operandTest{
	{"$0", "$0"},
	{"$256", "$256"},
	{"(R0)", "0(R0)"},
	{"(R11)", "0(R11)"},
	{"(g)", "0(R10)"}, // TODO: Should print 0(g).
	{"-12(R4)", "-12(R4)"},
	{"0(PC)", "0(PC)"},
	{"1024", "1024"},
	{"12(R(1))", "12(R1)"},
	{"12(R13)", "12(R13)"},
	{"R0", "R0"},
	{"R0->(32-1)", "R0->31"},
	{"R0<<R1", "R0<<R1"},
	{"R0>>R(1)", "R0>>R1"},
	{"R0@>(32-1)", "R0@>31"},
	{"R1", "R1"},
	{"R11", "R11"},
	{"R12", "R12"},
	{"R13", "R13"},
	{"R14", "R14"},
	{"R15", "R15"},
	{"R1<<2(R3)", "R1<<2(R3)"},
	{"R(1)<<2(R(3))", "R1<<2(R3)"},
	{"R2", "R2"},
	{"R3", "R3"},
	{"R4", "R4"},
	{"R(4)", "R4"},
	{"R5", "R5"},
	{"R6", "R6"},
	{"R7", "R7"},
	{"R8", "R8"},
	// TODO: Fix Dconv to handle these. MOVM print shows the registers.
	{"[R0,R1,g,R15]", "$33795"},
	{"[R0-R7]", "$255"},
	{"[R(0)-R(7)]", "$255"},
	{"[R0]", "$1"},
	{"[R1-R12]", "$8190"},
	{"armCAS64(SB)", "armCAS64+0(SB)"},
	{"asmcgocall<>(SB)", "asmcgocall<>+0(SB)"},
	{"c+28(FP)", "c+28(FP)"},
	{"g", "R10"}, // TODO: Should print g.
	{"gosave<>(SB)", "gosave<>+0(SB)"},
	{"retlo+12(FP)", "retlo+12(FP)"},
	{"runtime·_sfloat2(SB)", "runtime._sfloat2+0(SB)"},
	{"·AddUint32(SB)", "\"\".AddUint32+0(SB)"},
}

var ppc64OperandTests = []operandTest{
	{"$((1<<63)-1)", "$0x7fffffffffffffff"},
	{"$(-64*1024)", "$-65536"},
	{"$(1024 * 8)", "$8192"},
	{"$-1", "$-1"},
	{"$-24(R4)", "$-24(R4)"},
	{"$0", "$0"},
	{"$0(R1)", "$0(R1)"},
	{"$0.5", "$0.5"},
	{"$0x7000", "$28672"},
	{"$0x88888eef", "$0x88888eef"},
	{"$1", "$1"},
	{"$_main<>(SB)", "$_main<>+0(SB)"},
	{"$argframe+0(FP)", "$argframe+0(FP)"},
	{"$runtime·tlsg(SB)", "$runtime.tlsg(SB)"},
	{"$~3", "$-4"},
	{"(-288-3*8)(R1)", "-312(R1)"},
	{"(16)(R7)", "16(R7)"},
	{"(8)(g)", "8(R30)"}, // TODO: Should print 8(g)
	{"(CTR)", "0(CTR)"},
	{"(R0)", "0(R0)"},
	{"(R3)", "0(R3)"},
	{"(R4)", "0(R4)"},
	{"(R5)", "0(R5)"},
	{"-1(R4)", "-1(R4)"},
	{"-1(R5)", "-1(R5)"},
	{"6(PC)", "6(APC)"}, // TODO: Should print 6(PC).
	{"CR7", "C7"},       // TODO: Should print CR7.
	{"CTR", "CTR"},
	{"F14", "F14"},
	{"F15", "F15"},
	{"F16", "F16"},
	{"F17", "F17"},
	{"F18", "F18"},
	{"F19", "F19"},
	{"F20", "F20"},
	{"F21", "F21"},
	{"F22", "F22"},
	{"F23", "F23"},
	{"F24", "F24"},
	{"F25", "F25"},
	{"F26", "F26"},
	{"F27", "F27"},
	{"F28", "F28"},
	{"F29", "F29"},
	{"F30", "F30"},
	{"F31", "F31"},
	{"LR", "LR"},
	{"R0", "R0"},
	{"R1", "R1"},
	{"R11", "R11"},
	{"R12", "R12"},
	{"R13", "R13"},
	{"R14", "R14"},
	{"R15", "R15"},
	{"R16", "R16"},
	{"R17", "R17"},
	{"R18", "R18"},
	{"R19", "R19"},
	{"R2", "R2"},
	{"R20", "R20"},
	{"R21", "R21"},
	{"R22", "R22"},
	{"R23", "R23"},
	{"R24", "R24"},
	{"R25", "R25"},
	{"R26", "R26"},
	{"R27", "R27"},
	{"R28", "R28"},
	{"R29", "R29"},
	{"R3", "R3"},
	{"R31", "R31"},
	{"R4", "R4"},
	{"R5", "R5"},
	{"R6", "R6"},
	{"R7", "R7"},
	{"R8", "R8"},
	{"R9", "R9"},
	{"SPR(269)", "SPR(269)"},
	{"a+0(FP)", "a+0(FP)"},
	{"g", "R30"}, // TODO: Should print g.
	{"ret+8(FP)", "ret+8(FP)"},
	{"runtime·abort(SB)", "runtime.abort(SB)"},
	{"·AddUint32(SB)", "\"\".AddUint32(SB)"},
	{"·trunc(SB)", "\"\".trunc(SB)"},
}
