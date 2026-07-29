// asmcheck

// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package codegen

func booliface() interface{} {
	// 386:`LEAL\sruntime.staticuint64s\+8\(SB\)`
	// amd64:`LEAQ\sruntime.staticuint64s\+8\(SB\)`
	// mips:`MOVW\s[$]runtime.staticuint64s\+15\(SB\),\sR\d+`
	// mips64:`MOVV\s[$]runtime.staticuint64s\+15\(SB\),\sR\d+`
	return true
}

func smallint8iface() interface{} {
	// 386:`LEAL\sruntime.staticuint64s\+2024\(SB\)`
	// amd64:`LEAQ\sruntime.staticuint64s\+2024\(SB\)`
	// mips:`MOVW\s[$]runtime.staticuint64s\+2031\(SB\)`
	// mips64:`MOVV\s[$]runtime.staticuint64s\+2031\(SB\)`
	return int8(-3)
}

func smalluint8iface() interface{} {
	// 386:`LEAL\sruntime.staticuint64s\+24\(SB\)`
	// amd64:`LEAQ\sruntime.staticuint64s\+24\(SB\)`
	// mips:`MOVW\s[$]runtime.staticuint64s\+31\(SB\),\sR\d+`
	// mips64:`MOVV\s[$]runtime.staticuint64s\+31\(SB\),\sR\d+`
	return uint8(3)
}

func smallintiface() interface{} {
	// 386:`LEAL\sruntime.staticuint64s\+8\(SB\)`
	// amd64:`LEAQ\sruntime.staticuint64s\+8\(SB\)`
	// mips:`MOVW\s[$]runtime.staticuint64s\+12\(SB\),\sR\d+`
	// mips64:`MOVV\s[$]runtime.staticuint64s\+8\(SB\),\sR\d+`
	return 1
}

func smallint16iface() interface{} {
	// 386:`LEAL\sruntime.staticuint64s\+1016\(SB\)`
	// amd64:`LEAQ\sruntime.staticuint64s\+1016\(SB\)`
	// mips:`MOVW\s[$]runtime.staticuint64s\+1022\(SB\),\sR\d+`
	// mips64:`MOVV\s[$]runtime.staticuint64s\+1022\(SB\),\sR\d+`
	return int16(127)
}

func smallint32iface() interface{} {
	// 386:`LEAL\sruntime.staticuint64s\+2040\(SB\)`
	// amd64:`LEAQ\sruntime.staticuint64s\+2040\(SB\)`
	// mips:`MOVW\s[$]runtime.staticuint64s\+2044\(SB\),\sR\d+`
	// mips64:`MOVV\s[$]runtime.staticuint64s\+2044\(SB\),\sR\d+`
	return int32(255)
}

func smallint64iface() interface{} {
	// 386:`LEAL\sruntime.staticuint64s\+2040\(SB\)`
	// amd64:`LEAQ\sruntime.staticuint64s\+2040\(SB\)`
	// mips:`MOVW\s[$]runtime.staticuint64s\+2040\(SB\),\sR\d+`
	// mips64:`MOVV\s[$]runtime.staticuint64s\+2040\(SB\),\sR\d+`
	return int64(255)
}

func smalluintface() interface{} {
	// 386:`LEAL\sruntime.staticuint64s\+16\(SB\)`
	// amd64:`LEAQ\sruntime.staticuint64s\+16\(SB\)`
	// mips:`MOVW\s[$]runtime.staticuint64s\+20\(SB\),\sR\d+`
	// mips64:`MOVV\s[$]runtime.staticuint64s\+16\(SB\),\sR\d+`
	return uint(2)
}

func smalluintptriface() interface{} {
	// 386:`LEAL\sruntime.staticuint64s\+24\(SB\)`
	// amd64:`LEAQ\sruntime.staticuint64s\+24\(SB\)`
	// mips:`MOVW\s[$]runtime.staticuint64s\+28\(SB\),\sR\d+`
	// mips64:`MOVV\s[$]runtime.staticuint64s\+24\(SB\),\sR\d+`
	return uintptr(3)
}

func smalluint16iface() interface{} {
	// 386:`LEAL\sruntime.staticuint64s\+80\(SB\)`
	// amd64:`LEAQ\sruntime.staticuint64s\+80\(SB\)`
	// mips:`MOVW\s[$]runtime.staticuint64s\+86\(SB\),\sR\d+`
	// mips64:`MOVV\s[$]runtime.staticuint64s\+86\(SB\),\sR\d+`
	return uint16(10)
}

func smalluint32iface() interface{} {
	// 386:`LEAL\sruntime.staticuint64s\+8\(SB\)`
	// amd64:`LEAQ\sruntime.staticuint64s\+8\(SB\)`
	// mips:`MOVW\s[$]runtime.staticuint64s\+12\(SB\),\sR\d+`
	// mips64:`MOVV\s[$]runtime.staticuint64s\+12\(SB\),\sR\d+`
	return uint32(1)
}

func smalluint64iface() interface{} {
	// 386:`LEAL\sruntime.staticuint64s\+56\(SB\)`
	// amd64:`LEAQ\sruntime.staticuint64s\+56\(SB\)`
	// mips:`MOVW\s[$]runtime.staticuint64s\+56\(SB\),\sR\d+`
	// mips64:`MOVV\s[$]runtime.staticuint64s\+56\(SB\),\sR\d+`
	return uint64(7)
}
