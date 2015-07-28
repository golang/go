// Copyright 2015 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import "strings"

// copied from ../../amd64/reg.go
var regNamesAMD64 = []string{
	".AX",
	".CX",
	".DX",
	".BX",
	".SP",
	".BP",
	".SI",
	".DI",
	".R8",
	".R9",
	".R10",
	".R11",
	".R12",
	".R13",
	".R14",
	".R15",
	".X0",
	".X1",
	".X2",
	".X3",
	".X4",
	".X5",
	".X6",
	".X7",
	".X8",
	".X9",
	".X10",
	".X11",
	".X12",
	".X13",
	".X14",
	".X15",

	// pseudo-registers
	".SB",
	".FLAGS",
}

func init() {
	// Make map from reg names to reg integers.
	if len(regNamesAMD64) > 64 {
		panic("too many registers")
	}
	num := map[string]int{}
	for i, name := range regNamesAMD64 {
		if name[0] != '.' {
			panic("register name " + name + " does not start with '.'")
		}
		num[name[1:]] = i
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

	gp := buildReg("AX CX DX BX BP SI DI R8 R9 R10 R11 R12 R13 R14 R15")
	gpsp := gp | buildReg("SP")
	gpspsb := gpsp | buildReg("SB")
	flags := buildReg("FLAGS")
	gp01 := regInfo{[]regMask{}, 0, []regMask{gp}}
	gp11 := regInfo{[]regMask{gpsp}, 0, []regMask{gp}}
	gp11sb := regInfo{[]regMask{gpspsb}, 0, []regMask{gp}}
	gp21 := regInfo{[]regMask{gpsp, gpsp}, 0, []regMask{gp}}
	gp21sb := regInfo{[]regMask{gpspsb, gpsp}, 0, []regMask{gp}}
	gp21shift := regInfo{[]regMask{gpsp, buildReg("CX")}, 0, []regMask{gp}}
	gp2flags := regInfo{[]regMask{gpsp, gpsp}, 0, []regMask{flags}}
	gp1flags := regInfo{[]regMask{gpsp}, 0, []regMask{flags}}
	flagsgp1 := regInfo{[]regMask{flags}, 0, []regMask{gp}}
	gpload := regInfo{[]regMask{gpspsb, 0}, 0, []regMask{gp}}
	gploadidx := regInfo{[]regMask{gpspsb, gpsp, 0}, 0, []regMask{gp}}
	gpstore := regInfo{[]regMask{gpspsb, gpsp, 0}, 0, nil}
	gpstoreconst := regInfo{[]regMask{gpspsb, 0}, 0, nil}
	gpstoreidx := regInfo{[]regMask{gpspsb, gpsp, gpsp, 0}, 0, nil}
	flagsgp := regInfo{[]regMask{flags}, 0, []regMask{gp}}
	cmov := regInfo{[]regMask{flags, gp, gp}, 0, []regMask{gp}}

	// Suffixes encode the bit width of various instructions.
	// Q = 64 bit, L = 32 bit, W = 16 bit, B = 8 bit

	// TODO: 2-address instructions.  Mark ops as needing matching input/output regs.
	var AMD64ops = []opData{
		// binary ops
		{name: "ADDQ", reg: gp21, asm: "ADDQ"},      // arg0 + arg1
		{name: "ADDL", reg: gp21, asm: "ADDL"},      // arg0 + arg1
		{name: "ADDW", reg: gp21, asm: "ADDW"},      // arg0 + arg1
		{name: "ADDB", reg: gp21, asm: "ADDB"},      // arg0 + arg1
		{name: "ADDQconst", reg: gp11, asm: "ADDQ"}, // arg0 + auxint
		{name: "ADDLconst", reg: gp11, asm: "ADDL"}, // arg0 + auxint
		{name: "ADDWconst", reg: gp11, asm: "ADDW"}, // arg0 + auxint
		{name: "ADDBconst", reg: gp11, asm: "ADDB"}, // arg0 + auxint

		{name: "SUBQ", reg: gp21, asm: "SUBQ"},      // arg0 - arg1
		{name: "SUBL", reg: gp21, asm: "SUBL"},      // arg0 - arg1
		{name: "SUBW", reg: gp21, asm: "SUBW"},      // arg0 - arg1
		{name: "SUBB", reg: gp21, asm: "SUBB"},      // arg0 - arg1
		{name: "SUBQconst", reg: gp11, asm: "SUBQ"}, // arg0 - auxint
		{name: "SUBLconst", reg: gp11, asm: "SUBL"}, // arg0 - auxint
		{name: "SUBWconst", reg: gp11, asm: "SUBW"}, // arg0 - auxint
		{name: "SUBBconst", reg: gp11, asm: "SUBB"}, // arg0 - auxint

		{name: "MULQ", reg: gp21, asm: "IMULQ"},      // arg0 * arg1
		{name: "MULL", reg: gp21, asm: "IMULL"},      // arg0 * arg1
		{name: "MULW", reg: gp21, asm: "IMULW"},      // arg0 * arg1
		{name: "MULQconst", reg: gp11, asm: "IMULQ"}, // arg0 * auxint
		{name: "MULLconst", reg: gp11, asm: "IMULL"}, // arg0 * auxint
		{name: "MULWconst", reg: gp11, asm: "IMULW"}, // arg0 * auxint

		{name: "ANDQ", reg: gp21, asm: "ANDQ"},      // arg0 & arg1
		{name: "ANDL", reg: gp21, asm: "ANDL"},      // arg0 & arg1
		{name: "ANDW", reg: gp21, asm: "ANDW"},      // arg0 & arg1
		{name: "ANDB", reg: gp21, asm: "ANDB"},      // arg0 & arg1
		{name: "ANDQconst", reg: gp11, asm: "ANDQ"}, // arg0 & auxint
		{name: "ANDLconst", reg: gp11, asm: "ANDL"}, // arg0 & auxint
		{name: "ANDWconst", reg: gp11, asm: "ANDW"}, // arg0 & auxint
		{name: "ANDBconst", reg: gp11, asm: "ANDB"}, // arg0 & auxint

		{name: "ORQ", reg: gp21, asm: "ORQ"},      // arg0 | arg1
		{name: "ORL", reg: gp21, asm: "ORL"},      // arg0 | arg1
		{name: "ORW", reg: gp21, asm: "ORW"},      // arg0 | arg1
		{name: "ORB", reg: gp21, asm: "ORB"},      // arg0 | arg1
		{name: "ORQconst", reg: gp11, asm: "ORQ"}, // arg0 | auxint
		{name: "ORLconst", reg: gp11, asm: "ORL"}, // arg0 | auxint
		{name: "ORWconst", reg: gp11, asm: "ORW"}, // arg0 | auxint
		{name: "ORBconst", reg: gp11, asm: "ORB"}, // arg0 | auxint

		{name: "XORQ", reg: gp21, asm: "XORQ"},      // arg0 ^ arg1
		{name: "XORL", reg: gp21, asm: "XORL"},      // arg0 ^ arg1
		{name: "XORW", reg: gp21, asm: "XORW"},      // arg0 ^ arg1
		{name: "XORB", reg: gp21, asm: "XORB"},      // arg0 ^ arg1
		{name: "XORQconst", reg: gp11, asm: "XORQ"}, // arg0 ^ auxint
		{name: "XORLconst", reg: gp11, asm: "XORL"}, // arg0 ^ auxint
		{name: "XORWconst", reg: gp11, asm: "XORW"}, // arg0 ^ auxint
		{name: "XORBconst", reg: gp11, asm: "XORB"}, // arg0 ^ auxint

		{name: "CMPQ", reg: gp2flags, asm: "CMPQ"},      // arg0 compare to arg1
		{name: "CMPL", reg: gp2flags, asm: "CMPL"},      // arg0 compare to arg1
		{name: "CMPW", reg: gp2flags, asm: "CMPW"},      // arg0 compare to arg1
		{name: "CMPB", reg: gp2flags, asm: "CMPB"},      // arg0 compare to arg1
		{name: "CMPQconst", reg: gp1flags, asm: "CMPQ"}, // arg0 compare to auxint
		{name: "CMPLconst", reg: gp1flags, asm: "CMPL"}, // arg0 compare to auxint
		{name: "CMPWconst", reg: gp1flags, asm: "CMPW"}, // arg0 compare to auxint
		{name: "CMPBconst", reg: gp1flags, asm: "CMPB"}, // arg0 compare to auxint

		{name: "TESTQ", reg: gp2flags, asm: "TESTQ"},      // (arg0 & arg1) compare to 0
		{name: "TESTL", reg: gp2flags, asm: "TESTL"},      // (arg0 & arg1) compare to 0
		{name: "TESTW", reg: gp2flags, asm: "TESTW"},      // (arg0 & arg1) compare to 0
		{name: "TESTB", reg: gp2flags, asm: "TESTB"},      // (arg0 & arg1) compare to 0
		{name: "TESTQconst", reg: gp1flags, asm: "TESTQ"}, // (arg0 & auxint) compare to 0
		{name: "TESTLconst", reg: gp1flags, asm: "TESTL"}, // (arg0 & auxint) compare to 0
		{name: "TESTWconst", reg: gp1flags, asm: "TESTW"}, // (arg0 & auxint) compare to 0
		{name: "TESTBconst", reg: gp1flags, asm: "TESTB"}, // (arg0 & auxint) compare to 0

		{name: "SHLQ", reg: gp21shift, asm: "SHLQ"}, // arg0 << arg1, shift amount is mod 64
		{name: "SHLL", reg: gp21shift, asm: "SHLL"}, // arg0 << arg1, shift amount is mod 32
		{name: "SHLW", reg: gp21shift, asm: "SHLW"}, // arg0 << arg1, shift amount is mod 32
		{name: "SHLB", reg: gp21shift, asm: "SHLB"}, // arg0 << arg1, shift amount is mod 32
		{name: "SHLQconst", reg: gp11, asm: "SHLQ"}, // arg0 << auxint, shift amount 0-63
		{name: "SHLLconst", reg: gp11, asm: "SHLL"}, // arg0 << auxint, shift amount 0-31
		{name: "SHLWconst", reg: gp11, asm: "SHLW"}, // arg0 << auxint, shift amount 0-31
		{name: "SHLBconst", reg: gp11, asm: "SHLB"}, // arg0 << auxint, shift amount 0-31
		// Note: x86 is weird, the 16 and 8 byte shifts still use all 5 bits of shift amount!

		{name: "SHRQ", reg: gp21shift, asm: "SHRQ"}, // unsigned arg0 >> arg1, shift amount is mod 64
		{name: "SHRL", reg: gp21shift, asm: "SHRL"}, // unsigned arg0 >> arg1, shift amount is mod 32
		{name: "SHRW", reg: gp21shift, asm: "SHRW"}, // unsigned arg0 >> arg1, shift amount is mod 32
		{name: "SHRB", reg: gp21shift, asm: "SHRB"}, // unsigned arg0 >> arg1, shift amount is mod 32
		{name: "SHRQconst", reg: gp11, asm: "SHRQ"}, // unsigned arg0 >> auxint, shift amount 0-63
		{name: "SHRLconst", reg: gp11, asm: "SHRL"}, // unsigned arg0 >> auxint, shift amount 0-31
		{name: "SHRWconst", reg: gp11, asm: "SHRW"}, // unsigned arg0 >> auxint, shift amount 0-31
		{name: "SHRBconst", reg: gp11, asm: "SHRB"}, // unsigned arg0 >> auxint, shift amount 0-31

		{name: "SARQ", reg: gp21shift, asm: "SARQ"}, // signed arg0 >> arg1, shift amount is mod 64
		{name: "SARL", reg: gp21shift, asm: "SARL"}, // signed arg0 >> arg1, shift amount is mod 32
		{name: "SARW", reg: gp21shift, asm: "SARW"}, // signed arg0 >> arg1, shift amount is mod 32
		{name: "SARB", reg: gp21shift, asm: "SARB"}, // signed arg0 >> arg1, shift amount is mod 32
		{name: "SARQconst", reg: gp11, asm: "SARQ"}, // signed arg0 >> auxint, shift amount 0-63
		{name: "SARLconst", reg: gp11, asm: "SARL"}, // signed arg0 >> auxint, shift amount 0-31
		{name: "SARWconst", reg: gp11, asm: "SARW"}, // signed arg0 >> auxint, shift amount 0-31
		{name: "SARBconst", reg: gp11, asm: "SARB"}, // signed arg0 >> auxint, shift amount 0-31

		// unary ops
		{name: "NEGQ", reg: gp11, asm: "NEGQ"}, // -arg0
		{name: "NEGL", reg: gp11, asm: "NEGL"}, // -arg0
		{name: "NEGW", reg: gp11, asm: "NEGW"}, // -arg0
		{name: "NEGB", reg: gp11, asm: "NEGB"}, // -arg0

		{name: "SBBQcarrymask", reg: flagsgp1, asm: "SBBQ"}, // (int64)(-1) if carry is set, 0 if carry is clear.

		{name: "SETEQ", reg: flagsgp, asm: "SETEQ"}, // extract == condition from arg0
		{name: "SETNE", reg: flagsgp, asm: "SETNE"}, // extract != condition from arg0
		{name: "SETL", reg: flagsgp, asm: "SETLT"},  // extract signed < condition from arg0
		{name: "SETLE", reg: flagsgp, asm: "SETLE"}, // extract signed <= condition from arg0
		{name: "SETG", reg: flagsgp, asm: "SETGT"},  // extract signed > condition from arg0
		{name: "SETGE", reg: flagsgp, asm: "SETGE"}, // extract signed >= condition from arg0
		{name: "SETB", reg: flagsgp, asm: "SETCS"},  // extract unsigned < condition from arg0
		{name: "SETBE", reg: flagsgp, asm: "SETLS"}, // extract unsigned <= condition from arg0
		{name: "SETA", reg: flagsgp, asm: "SETHI"},  // extract unsigned > condition from arg0
		{name: "SETAE", reg: flagsgp, asm: "SETCC"}, // extract unsigned >= condition from arg0

		{name: "CMOVQCC", reg: cmov}, // carry clear

		{name: "MOVBQSX", reg: gp11, asm: "MOVBQSX"}, // sign extend arg0 from int8 to int64
		{name: "MOVBQZX", reg: gp11, asm: "MOVBQZX"}, // zero extend arg0 from int8 to int64
		{name: "MOVWQSX", reg: gp11, asm: "MOVWQSX"}, // sign extend arg0 from int16 to int64
		{name: "MOVWQZX", reg: gp11, asm: "MOVWQZX"}, // zero extend arg0 from int16 to int64
		{name: "MOVLQSX", reg: gp11, asm: "MOVLQSX"}, // sign extend arg0 from int32 to int64
		{name: "MOVLQZX", reg: gp11, asm: "MOVLQZX"}, // zero extend arg0 from int32 to int64

		{name: "MOVBconst", reg: gp01, asm: "MOVB"}, // 8 low bits of auxint
		{name: "MOVWconst", reg: gp01, asm: "MOVW"}, // 16 low bits of auxint
		{name: "MOVLconst", reg: gp01, asm: "MOVL"}, // 32 low bits of auxint
		{name: "MOVQconst", reg: gp01, asm: "MOVQ"}, // auxint

		{name: "LEAQ", reg: gp11sb},  // arg0 + auxint + offset encoded in aux
		{name: "LEAQ1", reg: gp21sb}, // arg0 + arg1 + auxint
		{name: "LEAQ2", reg: gp21sb}, // arg0 + 2*arg1 + auxint
		{name: "LEAQ4", reg: gp21sb}, // arg0 + 4*arg1 + auxint
		{name: "LEAQ8", reg: gp21sb}, // arg0 + 8*arg1 + auxint

		{name: "MOVBload", reg: gpload, asm: "MOVB"},          // load byte from arg0+auxint. arg1=mem
		{name: "MOVBQSXload", reg: gpload, asm: "MOVBQSX"},    // ditto, extend to int64
		{name: "MOVBQZXload", reg: gpload, asm: "MOVBQZX"},    // ditto, extend to uint64
		{name: "MOVWload", reg: gpload, asm: "MOVW"},          // load 2 bytes from arg0+auxint. arg1=mem
		{name: "MOVLload", reg: gpload, asm: "MOVL"},          // load 4 bytes from arg0+auxint. arg1=mem
		{name: "MOVQload", reg: gpload, asm: "MOVQ"},          // load 8 bytes from arg0+auxint. arg1=mem
		{name: "MOVQloadidx8", reg: gploadidx, asm: "MOVQ"},   // load 8 bytes from arg0+8*arg1+auxint. arg2=mem
		{name: "MOVBstore", reg: gpstore, asm: "MOVB"},        // store byte in arg1 to arg0+auxint. arg2=mem
		{name: "MOVWstore", reg: gpstore, asm: "MOVW"},        // store 2 bytes in arg1 to arg0+auxint. arg2=mem
		{name: "MOVLstore", reg: gpstore, asm: "MOVL"},        // store 4 bytes in arg1 to arg0+auxint. arg2=mem
		{name: "MOVQstore", reg: gpstore, asm: "MOVQ"},        // store 8 bytes in arg1 to arg0+auxint. arg2=mem
		{name: "MOVQstoreidx8", reg: gpstoreidx, asm: "MOVQ"}, // store 8 bytes in arg2 to arg0+8*arg1+auxint. arg3=mem

		{name: "MOVXzero", reg: gpstoreconst}, // store auxint 0 bytes into arg0 using a series of MOV instructions. arg1=mem.
		// TODO: implement this when register clobbering works
		{name: "REPSTOSQ", reg: regInfo{[]regMask{buildReg("DI"), buildReg("CX")}, buildReg("DI AX CX"), nil}}, // store arg1 8-byte words containing zero into arg0 using STOSQ. arg2=mem.

		//TODO: set register clobber to everything?
		{name: "CALLstatic"},                                                            // call static function aux.(*gc.Sym).  arg0=mem, returns mem
		{name: "CALLclosure", reg: regInfo{[]regMask{gpsp, buildReg("DX"), 0}, 0, nil}}, // call function via closure.  arg0=codeptr, arg1=closure, arg2=mem returns mem

		{name: "REPMOVSB", reg: regInfo{[]regMask{buildReg("DI"), buildReg("SI"), buildReg("CX")}, buildReg("DI SI CX"), nil}}, // move arg2 bytes from arg1 to arg0.  arg3=mem, returns memory

		// (InvertFlags (CMPQ a b)) == (CMPQ b a)
		// So if we want (SETL (CMPQ a b)) but we can't do that because a is a constant,
		// then we do (SETL (InvertFlags (CMPQ b a))) instead.
		// Rewrites will convert this to (SETG (CMPQ b a)).
		// InvertFlags is a pseudo-op which can't appear in assembly output.
		{name: "InvertFlags"}, // reverse direction of arg0
	}

	var AMD64blocks = []blockData{
		{name: "EQ"},
		{name: "NE"},
		{name: "LT"},
		{name: "LE"},
		{name: "GT"},
		{name: "GE"},
		{name: "ULT"},
		{name: "ULE"},
		{name: "UGT"},
		{name: "UGE"},
	}

	archs = append(archs, arch{"AMD64", AMD64ops, AMD64blocks, regNamesAMD64})
}
