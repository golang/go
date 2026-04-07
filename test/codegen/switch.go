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

// use jump tables for 8+ string cases
// Using multiple return values prevent lookup tables.
func squareJump(x int) (int, int) {
	// amd64:`JMP \(.*\)\(.*\)$`
	// arm64:`MOVD \(R.*\)\(R.*<<3\)` `JMP \(R.*\)$`
	// loong64: `ALSLV` `MOVV` `JMP`
	switch x {
	case 1:
		return 1, 1
	case 2:
		return 4, 2
	case 3:
		return 9, 3
	case 4:
		return 16, 4
	case 5:
		return 25, 5
	case 6:
		return 36, 6
	case 7:
		return 49, 7
	case 8:
		return 64, 8
	default:
		return x * x, x
	}
}

// use lookup tables for 8+ int cases returning constants
func squareLookup(x int) int {
	// amd64:`LEAQ .*\(SB\)` `MOVQ .*\(.*\)\(.*\*8\)` -`JMP \(.*\)\(.*\)$`
	// arm64:`MOVD \(R.*\)\(R.*<<3\)` -`JMP \(R.*\)$`
	// loong64:`SLLV` `MOVV \(R.*\)\(R.*\)` -`ALSLV`
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

// use lookup tables for 8+ bool-returning cases
func boolLookup(x int) bool {
	// amd64:`LEAQ .*\(SB\)` `MOVBLZX .*\(.*\)` -`JMP \(.*\)\(.*\)$`
	// arm64:`MOVBU \(R.*\)\(R.*\)` -`JMP \(R.*\)$`
	// loong64:`MOVBU \(R.*\)\(R.*\)` -`ALSLV`
	switch x {
	case 1:
		return true
	case 2:
		return false
	case 3:
		return true
	case 4:
		return false
	case 5:
		return true
	case 6:
		return false
	case 7:
		return true
	case 8:
		return false
	default:
		return x > 0
	}
}

// use lookup tables for 8+ float64-returning cases
func floatLookup(x int) float64 {
	// amd64:`LEAQ .*\(SB\)` `MOVSD .*\(.*\)\(.*\*8\)` -`JMP \(.*\)\(.*\)$`
	// arm64:`FMOVD \(R.*\)\(R.*<<3\)` -`JMP \(R.*\)$`
	// loong64:`SLLV` `MOVD \(R.*\)\(R.*\)` -`ALSLV`
	switch x {
	case 1:
		return 1.5
	case 2:
		return 2.5
	case 3:
		return 3.5
	case 4:
		return 4.5
	case 5:
		return 5.5
	case 6:
		return 6.5
	case 7:
		return 7.5
	case 8:
		return 8.5
	default:
		return 0.0
	}
}

// use lookup tables for 8+ string-returning cases
func stringLookup(x int) string {
	// amd64:`LEAQ .*\(SB\)` -`JMP \(.*\)\(.*\)$`
	// arm64:-`JMP \(R.*\)$`
	// loong64:-`ALSLV`
	switch x {
	case 1:
		return "a"
	case 2:
		return "b"
	case 3:
		return "c"
	case 4:
		return "d"
	case 5:
		return "e"
	case 6:
		return "f"
	case 7:
		return "g"
	case 8:
		return "h"
	default:
		return ""
	}
}

// use lookup tables for 8+ complex128-returning cases
func complexLookup(x int) complex128 {
	// amd64:`LEAQ .*\(SB\)` -`JMP \(.*\)\(.*\)$`
	// arm64:-`JMP \(R.*\)$`
	// loong64:-`ALSLV`
	switch x {
	case 1:
		return 1 + 2i
	case 2:
		return 3 + 4i
	case 3:
		return 5 + 6i
	case 4:
		return 7 + 8i
	case 5:
		return 9 + 10i
	case 6:
		return 11 + 12i
	case 7:
		return 13 + 14i
	case 8:
		return 15 + 16i
	default:
		return 0
	}
}

// use jump tables for 8+ string lengths
func length(x string) int {
	// amd64:`JMP \(.*\)\(.*\)$`
	// arm64:`MOVD \(R.*\)\(R.*<<3\)` `JMP \(R.*\)$`
	// loong64:`ALSLV` `MOVV` `JMP`
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
	// amd64: `CMPB 1\(.*\), \$104$` -`cmpstring`
	// arm64: `MOVB 1\(R.*\), R.*$` `CMPW \$104, R.*$` -`cmpstring`
	switch ext {
	// amd64: `CMPL \(.*\), \$1836345390$`
	// arm64: `MOVD \$1836345390` `CMPW R.*, R.*$`
	case ".htm":
		return "A"
	// amd64: `CMPL \(.*\), \$1953457454$`
	// arm64: `MOVD \$1953457454` `CMPW R.*, R.*$`
	case ".eot":
		return "B"
	// amd64: `CMPL \(.*\), \$1735815982$`
	// arm64: `MOVD \$1735815982` `CMPW R.*, R.*$`
	case ".svg":
		return "C"
	// amd64: `CMPL \(.*\), \$1718907950$`
	// arm64: `MOVD \$1718907950` `CMPW R.*, R.*$`
	case ".ttf":
		return "D"
	default:
		return ""
	}
}

// use jump tables for type switches to concrete types.
func typeSwitch(x any) int {
	// amd64:`JMP \(.*\)\(.*\)$`
	// arm64:`MOVD \(R.*\)\(R.*<<3\)` `JMP \(R.*\)$`
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
	// amd64:`CALL runtime.interfaceSwitch` `MOVL 16\(AX\)` `MOVQ 8\(.*\)(.*\*8)`
	// arm64:`CALL runtime.interfaceSwitch` `LDAR` `MOVWU 16\(R0\)` `MOVD \(R.*\)\(R.*\)`
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
	// amd64:`CALL runtime.interfaceSwitch` `MOVL 16\(AX\)` `MOVQ 8\(.*\)(.*\*8)`
	// arm64:`CALL runtime.interfaceSwitch` `LDAR` `MOVWU 16\(R0\)` `MOVD \(R.*\)\(R.*\)`
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
	// amd64:`CALL runtime.typeAssert` `MOVL 16\(AX\)` `MOVQ 8\(.*\)(.*\*1)`
	// arm64:`CALL runtime.typeAssert` `LDAR` `MOVWU 16\(R0\)` `MOVD \(R.*\)\(R.*\)`
	if _, ok := x.(I); ok {
		return 3
	}
	return 5
}

func interfaceCast2(x K) int {
	// amd64:`CALL runtime.typeAssert` `MOVL 16\(AX\)` `MOVQ 8\(.*\)(.*\*1)`
	// arm64:`CALL runtime.typeAssert` `LDAR` `MOVWU 16\(R0\)` `MOVD \(R.*\)\(R.*\)`
	if _, ok := x.(I); ok {
		return 3
	}
	return 5
}

func interfaceConv(x IJ) I {
	// amd64:`CALL runtime.typeAssert` `MOVL 16\(AX\)` `MOVQ 8\(.*\)(.*\*1)`
	// arm64:`CALL runtime.typeAssert` `LDAR` `MOVWU 16\(R0\)` `MOVD \(R.*\)\(R.*\)`
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
	// amd64:-"CMP" -"CALL"
	// arm64:-"CMP" -"CALL"
	stringSwitchInlineable("foo")
}
