// asmcheck

// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// These tests check code generation of switch statements.

package codegen

// see issue 33934
func f(x string) int {
	// amd64:-`cmpstring`
	switch x {
	case "":
		return -1
	case "1", "2", "3":
		return -2
	default:
		return -3
	}
}

// use jump tables for 8+ int cases
func square(x int) int {
	// amd64:`JMP\s\(.*\)\(.*\)$`
	// arm64:`MOVD\s\(R.*\)\(R.*<<3\)`,`JMP\s\(R.*\)$`
	// loong64: `ALSLV`,`MOVV`,`JMP`
	switch x {
	case 1:
		return 1
	case 2:
		return 4
	case 3:
		return 9
	case 4:
		return 16
	case 5:
		return 25
	case 6:
		return 36
	case 7:
		return 49
	case 8:
		return 64
	default:
		return x * x
	}
}

// use jump tables for 8+ string lengths
func length(x string) int {
	// amd64:`JMP\s\(.*\)\(.*\)$`
	// arm64:`MOVD\s\(R.*\)\(R.*<<3\)`,`JMP\s\(R.*\)$`
	// loong64:`ALSLV`,`MOVV`,`JMP`
	switch x {
	case "a":
		return 1
	case "bb":
		return 2
	case "ccc":
		return 3
	case "dddd":
		return 4
	case "eeeee":
		return 5
	case "ffffff":
		return 6
	case "ggggggg":
		return 7
	case "hhhhhhhh":
		return 8
	default:
		return len(x)
	}
}

// Use single-byte ordered comparisons for binary searching strings.
// See issue 53333.
func mimetype(ext string) string {
	// amd64: `CMPB\s1\(.*\), \$104$`,-`cmpstring`
	// arm64: `MOVB\s1\(R.*\), R.*$`, `CMPW\s\$104, R.*$`, -`cmpstring`
	switch ext {
	// amd64: `CMPL\s\(.*\), \$1836345390$`
	// arm64: `MOVD\s\$1836345390`, `CMPW\sR.*, R.*$`
	case ".htm":
		return "A"
	// amd64: `CMPL\s\(.*\), \$1953457454$`
	// arm64: `MOVD\s\$1953457454`, `CMPW\sR.*, R.*$`
	case ".eot":
		return "B"
	// amd64: `CMPL\s\(.*\), \$1735815982$`
	// arm64: `MOVD\s\$1735815982`, `CMPW\sR.*, R.*$`
	case ".svg":
		return "C"
	// amd64: `CMPL\s\(.*\), \$1718907950$`
	// arm64: `MOVD\s\$1718907950`, `CMPW\sR.*, R.*$`
	case ".ttf":
		return "D"
	default:
		return ""
	}
}

// use jump tables for type switches to concrete types.
func typeSwitch(x any) int {
	// amd64:`JMP\s\(.*\)\(.*\)$`
	// arm64:`MOVD\s\(R.*\)\(R.*<<3\)`,`JMP\s\(R.*\)$`
	switch x.(type) {
	case int:
		return 0
	case int8:
		return 1
	case int16:
		return 2
	case int32:
		return 3
	case int64:
		return 4
	}
	return 7
}

type I interface {
	foo()
}
type J interface {
	bar()
}
type IJ interface {
	I
	J
}
type K interface {
	baz()
}

// use a runtime call for type switches to interface types.
func interfaceSwitch(x any) int {
	// amd64:`CALL\truntime.interfaceSwitch`,`MOVL\t16\(AX\)`,`MOVQ\t8\(.*\)(.*\*8)`
	// arm64:`CALL\truntime.interfaceSwitch`,`LDAR`,`MOVWU\t16\(R0\)`,`MOVD\t\(R.*\)\(R.*\)`
	switch x.(type) {
	case I:
		return 1
	case J:
		return 2
	default:
		return 3
	}
}

func interfaceSwitch2(x K) int {
	// amd64:`CALL\truntime.interfaceSwitch`,`MOVL\t16\(AX\)`,`MOVQ\t8\(.*\)(.*\*8)`
	// arm64:`CALL\truntime.interfaceSwitch`,`LDAR`,`MOVWU\t16\(R0\)`,`MOVD\t\(R.*\)\(R.*\)`
	switch x.(type) {
	case I:
		return 1
	case J:
		return 2
	default:
		return 3
	}
}

func interfaceCast(x any) int {
	// amd64:`CALL\truntime.typeAssert`,`MOVL\t16\(AX\)`,`MOVQ\t8\(.*\)(.*\*1)`
	// arm64:`CALL\truntime.typeAssert`,`LDAR`,`MOVWU\t16\(R0\)`,`MOVD\t\(R.*\)\(R.*\)`
	if _, ok := x.(I); ok {
		return 3
	}
	return 5
}

func interfaceCast2(x K) int {
	// amd64:`CALL\truntime.typeAssert`,`MOVL\t16\(AX\)`,`MOVQ\t8\(.*\)(.*\*1)`
	// arm64:`CALL\truntime.typeAssert`,`LDAR`,`MOVWU\t16\(R0\)`,`MOVD\t\(R.*\)\(R.*\)`
	if _, ok := x.(I); ok {
		return 3
	}
	return 5
}

func interfaceConv(x IJ) I {
	// amd64:`CALL\truntime.typeAssert`,`MOVL\t16\(AX\)`,`MOVQ\t8\(.*\)(.*\*1)`
	// arm64:`CALL\truntime.typeAssert`,`LDAR`,`MOVWU\t16\(R0\)`,`MOVD\t\(R.*\)\(R.*\)`
	return x
}

// Make sure we can constant fold after inlining. See issue 71699.
func stringSwitchInlineable(s string) {
	switch s {
	case "foo", "bar", "baz", "goo":
	default:
		println("no")
	}
}
func stringSwitch() {
	// amd64:-"CMP",-"CALL"
	// arm64:-"CMP",-"CALL"
	stringSwitchInlineable("foo")
}
