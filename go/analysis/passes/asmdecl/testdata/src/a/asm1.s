// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build amd64

// Commented-out code should be ignored.
//
//	TEXT ·unknown(SB),0,$0
//		RET

TEXT ·arg1(SB),0,$0-2
	MOVB	x+0(FP), AX
	// MOVB x+0(FP), AX // commented out instructions used to panic
	MOVB	y+1(FP), BX
	MOVW	x+0(FP), AX // want `\[amd64\] arg1: invalid MOVW of x\+0\(FP\); int8 is 1-byte value`
	MOVW	y+1(FP), AX // want `invalid MOVW of y\+1\(FP\); uint8 is 1-byte value`
	MOVL	x+0(FP), AX // want `invalid MOVL of x\+0\(FP\); int8 is 1-byte value`
	MOVL	y+1(FP), AX // want `invalid MOVL of y\+1\(FP\); uint8 is 1-byte value`
	MOVQ	x+0(FP), AX // want `invalid MOVQ of x\+0\(FP\); int8 is 1-byte value`
	MOVQ	y+1(FP), AX // want `invalid MOVQ of y\+1\(FP\); uint8 is 1-byte value`
	MOVB	x+1(FP), AX // want `invalid offset x\+1\(FP\); expected x\+0\(FP\)`
	MOVB	y+2(FP), AX // want `invalid offset y\+2\(FP\); expected y\+1\(FP\)`
	TESTB	x+0(FP), AX
	TESTB	y+1(FP), BX
	TESTW	x+0(FP), AX // want `invalid TESTW of x\+0\(FP\); int8 is 1-byte value`
	TESTW	y+1(FP), AX // want `invalid TESTW of y\+1\(FP\); uint8 is 1-byte value`
	TESTL	x+0(FP), AX // want `invalid TESTL of x\+0\(FP\); int8 is 1-byte value`
	TESTL	y+1(FP), AX // want `invalid TESTL of y\+1\(FP\); uint8 is 1-byte value`
	TESTQ	x+0(FP), AX // want `invalid TESTQ of x\+0\(FP\); int8 is 1-byte value`
	TESTQ	y+1(FP), AX // want `invalid TESTQ of y\+1\(FP\); uint8 is 1-byte value`
	TESTB	x+1(FP), AX // want `invalid offset x\+1\(FP\); expected x\+0\(FP\)`
	TESTB	y+2(FP), AX // want `invalid offset y\+2\(FP\); expected y\+1\(FP\)`
	MOVB	8(SP), AX // want `8\(SP\) should be x\+0\(FP\)`
	MOVB	9(SP), AX // want `9\(SP\) should be y\+1\(FP\)`
	MOVB	10(SP), AX // want `use of 10\(SP\) points beyond argument frame`
	RET

TEXT ·arg2(SB),0,$0-4
	MOVB	x+0(FP), AX // want `arg2: invalid MOVB of x\+0\(FP\); int16 is 2-byte value`
	MOVB	y+2(FP), AX // want `invalid MOVB of y\+2\(FP\); uint16 is 2-byte value`
	MOVW	x+0(FP), AX
	MOVW	y+2(FP), BX
	MOVL	x+0(FP), AX // want `invalid MOVL of x\+0\(FP\); int16 is 2-byte value`
	MOVL	y+2(FP), AX // want `invalid MOVL of y\+2\(FP\); uint16 is 2-byte value`
	MOVQ	x+0(FP), AX // want `invalid MOVQ of x\+0\(FP\); int16 is 2-byte value`
	MOVQ	y+2(FP), AX // want `invalid MOVQ of y\+2\(FP\); uint16 is 2-byte value`
	MOVW	x+2(FP), AX // want `invalid offset x\+2\(FP\); expected x\+0\(FP\)`
	MOVW	y+0(FP), AX // want `invalid offset y\+0\(FP\); expected y\+2\(FP\)`
	TESTB	x+0(FP), AX // want `invalid TESTB of x\+0\(FP\); int16 is 2-byte value`
	TESTB	y+2(FP), AX // want `invalid TESTB of y\+2\(FP\); uint16 is 2-byte value`
	TESTW	x+0(FP), AX
	TESTW	y+2(FP), BX
	TESTL	x+0(FP), AX // want `invalid TESTL of x\+0\(FP\); int16 is 2-byte value`
	TESTL	y+2(FP), AX // want `invalid TESTL of y\+2\(FP\); uint16 is 2-byte value`
	TESTQ	x+0(FP), AX // want `invalid TESTQ of x\+0\(FP\); int16 is 2-byte value`
	TESTQ	y+2(FP), AX // want `invalid TESTQ of y\+2\(FP\); uint16 is 2-byte value`
	TESTW	x+2(FP), AX // want `invalid offset x\+2\(FP\); expected x\+0\(FP\)`
	TESTW	y+0(FP), AX // want `invalid offset y\+0\(FP\); expected y\+2\(FP\)`
	RET

TEXT ·arg4(SB),0,$0-2 // want `arg4: wrong argument size 2; expected \$\.\.\.-8`
	MOVB	x+0(FP), AX // want `invalid MOVB of x\+0\(FP\); int32 is 4-byte value`
	MOVB	y+4(FP), BX // want `invalid MOVB of y\+4\(FP\); uint32 is 4-byte value`
	MOVW	x+0(FP), AX // want `invalid MOVW of x\+0\(FP\); int32 is 4-byte value`
	MOVW	y+4(FP), AX // want `invalid MOVW of y\+4\(FP\); uint32 is 4-byte value`
	MOVL	x+0(FP), AX
	MOVL	y+4(FP), AX
	MOVQ	x+0(FP), AX // want `invalid MOVQ of x\+0\(FP\); int32 is 4-byte value`
	MOVQ	y+4(FP), AX // want `invalid MOVQ of y\+4\(FP\); uint32 is 4-byte value`
	MOVL	x+4(FP), AX // want `invalid offset x\+4\(FP\); expected x\+0\(FP\)`
	MOVL	y+2(FP), AX // want `invalid offset y\+2\(FP\); expected y\+4\(FP\)`
	TESTB	x+0(FP), AX // want `invalid TESTB of x\+0\(FP\); int32 is 4-byte value`
	TESTB	y+4(FP), BX // want `invalid TESTB of y\+4\(FP\); uint32 is 4-byte value`
	TESTW	x+0(FP), AX // want `invalid TESTW of x\+0\(FP\); int32 is 4-byte value`
	TESTW	y+4(FP), AX // want `invalid TESTW of y\+4\(FP\); uint32 is 4-byte value`
	TESTL	x+0(FP), AX
	TESTL	y+4(FP), AX
	TESTQ	x+0(FP), AX // want `invalid TESTQ of x\+0\(FP\); int32 is 4-byte value`
	TESTQ	y+4(FP), AX // want `invalid TESTQ of y\+4\(FP\); uint32 is 4-byte value`
	TESTL	x+4(FP), AX // want `invalid offset x\+4\(FP\); expected x\+0\(FP\)`
	TESTL	y+2(FP), AX // want `invalid offset y\+2\(FP\); expected y\+4\(FP\)`
	RET

TEXT ·arg8(SB),7,$0-2 // want `wrong argument size 2; expected \$\.\.\.-16`
	MOVB	x+0(FP), AX // want `invalid MOVB of x\+0\(FP\); int64 is 8-byte value`
	MOVB	y+8(FP), BX // want `invalid MOVB of y\+8\(FP\); uint64 is 8-byte value`
	MOVW	x+0(FP), AX // want `invalid MOVW of x\+0\(FP\); int64 is 8-byte value`
	MOVW	y+8(FP), AX // want `invalid MOVW of y\+8\(FP\); uint64 is 8-byte value`
	MOVL	x+0(FP), AX // want `invalid MOVL of x\+0\(FP\); int64 is 8-byte value`
	MOVL	y+8(FP), AX // want `invalid MOVL of y\+8\(FP\); uint64 is 8-byte value`
	MOVQ	x+0(FP), AX
	MOVQ	y+8(FP), AX
	MOVQ	x+8(FP), AX // want `invalid offset x\+8\(FP\); expected x\+0\(FP\)`
	MOVQ	y+2(FP), AX // want `invalid offset y\+2\(FP\); expected y\+8\(FP\)`
	TESTB	x+0(FP), AX // want `invalid TESTB of x\+0\(FP\); int64 is 8-byte value`
	TESTB	y+8(FP), BX // want `invalid TESTB of y\+8\(FP\); uint64 is 8-byte value`
	TESTW	x+0(FP), AX // want `invalid TESTW of x\+0\(FP\); int64 is 8-byte value`
	TESTW	y+8(FP), AX // want `invalid TESTW of y\+8\(FP\); uint64 is 8-byte value`
	TESTL	x+0(FP), AX // want `invalid TESTL of x\+0\(FP\); int64 is 8-byte value`
	TESTL	y+8(FP), AX // want `invalid TESTL of y\+8\(FP\); uint64 is 8-byte value`
	TESTQ	x+0(FP), AX
	TESTQ	y+8(FP), AX
	TESTQ	x+8(FP), AX // want `invalid offset x\+8\(FP\); expected x\+0\(FP\)`
	TESTQ	y+2(FP), AX // want `invalid offset y\+2\(FP\); expected y\+8\(FP\)`
	RET

TEXT ·argint(SB),0,$0-2 // want `wrong argument size 2; expected \$\.\.\.-16`
	MOVB	x+0(FP), AX // want `invalid MOVB of x\+0\(FP\); int is 8-byte value`
	MOVB	y+8(FP), BX // want `invalid MOVB of y\+8\(FP\); uint is 8-byte value`
	MOVW	x+0(FP), AX // want `invalid MOVW of x\+0\(FP\); int is 8-byte value`
	MOVW	y+8(FP), AX // want `invalid MOVW of y\+8\(FP\); uint is 8-byte value`
	MOVL	x+0(FP), AX // want `invalid MOVL of x\+0\(FP\); int is 8-byte value`
	MOVL	y+8(FP), AX // want `invalid MOVL of y\+8\(FP\); uint is 8-byte value`
	MOVQ	x+0(FP), AX
	MOVQ	y+8(FP), AX
	MOVQ	x+8(FP), AX // want `invalid offset x\+8\(FP\); expected x\+0\(FP\)`
	MOVQ	y+2(FP), AX // want `invalid offset y\+2\(FP\); expected y\+8\(FP\)`
	TESTB	x+0(FP), AX // want `invalid TESTB of x\+0\(FP\); int is 8-byte value`
	TESTB	y+8(FP), BX // want `invalid TESTB of y\+8\(FP\); uint is 8-byte value`
	TESTW	x+0(FP), AX // want `invalid TESTW of x\+0\(FP\); int is 8-byte value`
	TESTW	y+8(FP), AX // want `invalid TESTW of y\+8\(FP\); uint is 8-byte value`
	TESTL	x+0(FP), AX // want `invalid TESTL of x\+0\(FP\); int is 8-byte value`
	TESTL	y+8(FP), AX // want `invalid TESTL of y\+8\(FP\); uint is 8-byte value`
	TESTQ	x+0(FP), AX
	TESTQ	y+8(FP), AX
	TESTQ	x+8(FP), AX // want `invalid offset x\+8\(FP\); expected x\+0\(FP\)`
	TESTQ	y+2(FP), AX // want `invalid offset y\+2\(FP\); expected y\+8\(FP\)`
	RET

TEXT ·argptr(SB),7,$0-2 // want `wrong argument size 2; expected \$\.\.\.-40`
	MOVB	x+0(FP), AX // want `invalid MOVB of x\+0\(FP\); \*byte is 8-byte value`
	MOVB	y+8(FP), BX // want `invalid MOVB of y\+8\(FP\); \*byte is 8-byte value`
	MOVW	x+0(FP), AX // want `invalid MOVW of x\+0\(FP\); \*byte is 8-byte value`
	MOVW	y+8(FP), AX // want `invalid MOVW of y\+8\(FP\); \*byte is 8-byte value`
	MOVL	x+0(FP), AX // want `invalid MOVL of x\+0\(FP\); \*byte is 8-byte value`
	MOVL	y+8(FP), AX // want `invalid MOVL of y\+8\(FP\); \*byte is 8-byte value`
	MOVQ	x+0(FP), AX
	MOVQ	y+8(FP), AX
	MOVQ	x+8(FP), AX // want `invalid offset x\+8\(FP\); expected x\+0\(FP\)`
	MOVQ	y+2(FP), AX // want `invalid offset y\+2\(FP\); expected y\+8\(FP\)`
	TESTB	x+0(FP), AX // want `invalid TESTB of x\+0\(FP\); \*byte is 8-byte value`
	TESTB	y+8(FP), BX // want `invalid TESTB of y\+8\(FP\); \*byte is 8-byte value`
	TESTW	x+0(FP), AX // want `invalid TESTW of x\+0\(FP\); \*byte is 8-byte value`
	TESTW	y+8(FP), AX // want `invalid TESTW of y\+8\(FP\); \*byte is 8-byte value`
	TESTL	x+0(FP), AX // want `invalid TESTL of x\+0\(FP\); \*byte is 8-byte value`
	TESTL	y+8(FP), AX // want `invalid TESTL of y\+8\(FP\); \*byte is 8-byte value`
	TESTQ	x+0(FP), AX
	TESTQ	y+8(FP), AX
	TESTQ	x+8(FP), AX // want `invalid offset x\+8\(FP\); expected x\+0\(FP\)`
	TESTQ	y+2(FP), AX // want `invalid offset y\+2\(FP\); expected y\+8\(FP\)`
	MOVL	c+16(FP), AX // want `invalid MOVL of c\+16\(FP\); chan int is 8-byte value`
	MOVL	m+24(FP), AX // want `invalid MOVL of m\+24\(FP\); map\[int\]int is 8-byte value`
	MOVL	f+32(FP), AX // want `invalid MOVL of f\+32\(FP\); func\(\) is 8-byte value`
	RET

TEXT ·argstring(SB),0,$32 // want `wrong argument size 0; expected \$\.\.\.-32`
	MOVW	x+0(FP), AX // want `invalid MOVW of x\+0\(FP\); string base is 8-byte value`
	MOVL	x+0(FP), AX // want `invalid MOVL of x\+0\(FP\); string base is 8-byte value`
	LEAQ	x+0(FP), AX // ok
	MOVQ	x+0(FP), AX
	MOVW	x_base+0(FP), AX // want `invalid MOVW of x_base\+0\(FP\); string base is 8-byte value`
	MOVL	x_base+0(FP), AX // want `invalid MOVL of x_base\+0\(FP\); string base is 8-byte value`
	MOVQ	x_base+0(FP), AX
	MOVW	x_len+0(FP), AX // want `invalid offset x_len\+0\(FP\); expected x_len\+8\(FP\)`
	MOVL	x_len+0(FP), AX // want `invalid offset x_len\+0\(FP\); expected x_len\+8\(FP\)`
	MOVQ	x_len+0(FP), AX // want `invalid offset x_len\+0\(FP\); expected x_len\+8\(FP\)`
	MOVW	x_len+8(FP), AX // want `invalid MOVW of x_len\+8\(FP\); string len is 8-byte value`
	MOVL	x_len+8(FP), AX // want `invalid MOVL of x_len\+8\(FP\); string len is 8-byte value`
	MOVQ	x_len+8(FP), AX
	MOVQ	y+0(FP), AX // want `invalid offset y\+0\(FP\); expected y\+16\(FP\)`
	MOVQ	y_len+8(FP), AX // want `invalid offset y_len\+8\(FP\); expected y_len\+24\(FP\)`
	RET

TEXT ·argslice(SB),0,$48 // want `wrong argument size 0; expected \$\.\.\.-48`
	MOVW	x+0(FP), AX // want `invalid MOVW of x\+0\(FP\); slice base is 8-byte value`
	MOVL	x+0(FP), AX // want `invalid MOVL of x\+0\(FP\); slice base is 8-byte value`
	MOVQ	x+0(FP), AX
	MOVW	x_base+0(FP), AX // want `invalid MOVW of x_base\+0\(FP\); slice base is 8-byte value`
	MOVL	x_base+0(FP), AX // want `invalid MOVL of x_base\+0\(FP\); slice base is 8-byte value`
	MOVQ	x_base+0(FP), AX
	MOVW	x_len+0(FP), AX // want `invalid offset x_len\+0\(FP\); expected x_len\+8\(FP\)`
	MOVL	x_len+0(FP), AX // want `invalid offset x_len\+0\(FP\); expected x_len\+8\(FP\)`
	MOVQ	x_len+0(FP), AX // want `invalid offset x_len\+0\(FP\); expected x_len\+8\(FP\)`
	MOVW	x_len+8(FP), AX // want `invalid MOVW of x_len\+8\(FP\); slice len is 8-byte value`
	MOVL	x_len+8(FP), AX // want `invalid MOVL of x_len\+8\(FP\); slice len is 8-byte value`
	MOVQ	x_len+8(FP), AX
	MOVW	x_cap+0(FP), AX // want `invalid offset x_cap\+0\(FP\); expected x_cap\+16\(FP\)`
	MOVL	x_cap+0(FP), AX // want `invalid offset x_cap\+0\(FP\); expected x_cap\+16\(FP\)`
	MOVQ	x_cap+0(FP), AX // want `invalid offset x_cap\+0\(FP\); expected x_cap\+16\(FP\)`
	MOVW	x_cap+16(FP), AX // want `invalid MOVW of x_cap\+16\(FP\); slice cap is 8-byte value`
	MOVL	x_cap+16(FP), AX // want `invalid MOVL of x_cap\+16\(FP\); slice cap is 8-byte value`
	MOVQ	x_cap+16(FP), AX
	MOVQ	y+0(FP), AX // want `invalid offset y\+0\(FP\); expected y\+24\(FP\)`
	MOVQ	y_len+8(FP), AX // want `invalid offset y_len\+8\(FP\); expected y_len\+32\(FP\)`
	MOVQ	y_cap+16(FP), AX // want `invalid offset y_cap\+16\(FP\); expected y_cap\+40\(FP\)`
	RET

TEXT ·argiface(SB),0,$0-32
	MOVW	x+0(FP), AX // want `invalid MOVW of x\+0\(FP\); interface type is 8-byte value`
	MOVL	x+0(FP), AX // want `invalid MOVL of x\+0\(FP\); interface type is 8-byte value`
	MOVQ	x+0(FP), AX
	MOVW	x_type+0(FP), AX // want `invalid MOVW of x_type\+0\(FP\); interface type is 8-byte value`
	MOVL	x_type+0(FP), AX // want `invalid MOVL of x_type\+0\(FP\); interface type is 8-byte value`
	MOVQ	x_type+0(FP), AX
	MOVQ	x_itable+0(FP), AX // want `unknown variable x_itable; offset 0 is x_type\+0\(FP\)`
	MOVQ	x_itable+1(FP), AX // want `unknown variable x_itable; offset 1 is x_type\+0\(FP\)`
	MOVW	x_data+0(FP), AX // want `invalid offset x_data\+0\(FP\); expected x_data\+8\(FP\)`
	MOVL	x_data+0(FP), AX // want `invalid offset x_data\+0\(FP\); expected x_data\+8\(FP\)`
	MOVQ	x_data+0(FP), AX // want `invalid offset x_data\+0\(FP\); expected x_data\+8\(FP\)`
	MOVW	x_data+8(FP), AX // want `invalid MOVW of x_data\+8\(FP\); interface data is 8-byte value`
	MOVL	x_data+8(FP), AX // want `invalid MOVL of x_data\+8\(FP\); interface data is 8-byte value`
	MOVQ	x_data+8(FP), AX
	MOVW	y+16(FP), AX // want `invalid MOVW of y\+16\(FP\); interface itable is 8-byte value`
	MOVL	y+16(FP), AX // want `invalid MOVL of y\+16\(FP\); interface itable is 8-byte value`
	MOVQ	y+16(FP), AX
	MOVW	y_itable+16(FP), AX // want `invalid MOVW of y_itable\+16\(FP\); interface itable is 8-byte value`
	MOVL	y_itable+16(FP), AX // want `invalid MOVL of y_itable\+16\(FP\); interface itable is 8-byte value`
	MOVQ	y_itable+16(FP), AX
	MOVQ	y_type+16(FP), AX // want `unknown variable y_type; offset 16 is y_itable\+16\(FP\)`
	MOVW	y_data+16(FP), AX // want `invalid offset y_data\+16\(FP\); expected y_data\+24\(FP\)`
	MOVL	y_data+16(FP), AX // want `invalid offset y_data\+16\(FP\); expected y_data\+24\(FP\)`
	MOVQ	y_data+16(FP), AX // want `invalid offset y_data\+16\(FP\); expected y_data\+24\(FP\)`
	MOVW	y_data+24(FP), AX // want `invalid MOVW of y_data\+24\(FP\); interface data is 8-byte value`
	MOVL	y_data+24(FP), AX // want `invalid MOVL of y_data\+24\(FP\); interface data is 8-byte value`
	MOVQ	y_data+24(FP), AX
	RET

TEXT ·argcomplex(SB),0,$24 // want `wrong argument size 0; expected \$\.\.\.-24`
	MOVSS	x+0(FP), X0 // want `invalid MOVSS of x\+0\(FP\); complex64 is 8-byte value containing x_real\+0\(FP\) and x_imag\+4\(FP\)`
	MOVSS	x_real+0(FP), X0
	MOVSD	x_real+0(FP), X0 // want `invalid MOVSD of x_real\+0\(FP\); real\(complex64\) is 4-byte value`
	MOVSS	x_real+4(FP), X0 // want `invalid offset x_real\+4\(FP\); expected x_real\+0\(FP\)`
	MOVSS	x_imag+4(FP), X0
	MOVSD	x_imag+4(FP), X0 // want `invalid MOVSD of x_imag\+4\(FP\); imag\(complex64\) is 4-byte value`
	MOVSS	x_imag+8(FP), X0 // want `invalid offset x_imag\+8\(FP\); expected x_imag\+4\(FP\)`
	MOVSD	y+8(FP), X0 // want `invalid MOVSD of y\+8\(FP\); complex128 is 16-byte value containing y_real\+8\(FP\) and y_imag\+16\(FP\)`
	MOVSS	y_real+8(FP), X0 // want `invalid MOVSS of y_real\+8\(FP\); real\(complex128\) is 8-byte value`
	MOVSD	y_real+8(FP), X0
	MOVSS	y_real+16(FP), X0 // want `invalid offset y_real\+16\(FP\); expected y_real\+8\(FP\)`
	MOVSS	y_imag+16(FP), X0 // want `invalid MOVSS of y_imag\+16\(FP\); imag\(complex128\) is 8-byte value`
	MOVSD	y_imag+16(FP), X0
	MOVSS	y_imag+24(FP), X0 // want `invalid offset y_imag\+24\(FP\); expected y_imag\+16\(FP\)`
	// Loading both parts of a complex is ok: see issue 35264.
	MOVSD	x+0(FP), X0
	MOVO	y+8(FP), X0
	MOVOU	y+8(FP), X0
	// These are not ok.
	MOVO	x+0(FP), X0 // want `invalid MOVO of x\+0\(FP\); complex64 is 8-byte value containing x_real\+0\(FP\) and x_imag\+4\(FP\)`
	MOVOU	x+0(FP), X0 // want `invalid MOVOU of x\+0\(FP\); complex64 is 8-byte value containing x_real\+0\(FP\) and x_imag\+4\(FP\)`
	RET

TEXT ·argstruct(SB),0,$64 // want `wrong argument size 0; expected \$\.\.\.-24`
	MOVQ	x+0(FP), AX // want `invalid MOVQ of x\+0\(FP\); a.S is 24-byte value`
	MOVQ	x_i+0(FP), AX // want `invalid MOVQ of x_i\+0\(FP\); int32 is 4-byte value`
	MOVQ	x_b+0(FP), AX // want `invalid offset x_b\+0\(FP\); expected x_b\+4\(FP\)`
	MOVQ	x_s+8(FP), AX
	MOVQ	x_s_base+8(FP), AX
	MOVQ	x_s+16(FP), AX // want `invalid offset x_s\+16\(FP\); expected x_s\+8\(FP\), x_s_base\+8\(FP\), or x_s_len\+16\(FP\)`
	MOVQ	x_s_len+16(FP), AX
	RET

TEXT ·argarray(SB),0,$64 // want `wrong argument size 0; expected \$\.\.\.-48`
	MOVQ	x+0(FP), AX // want `invalid MOVQ of x\+0\(FP\); \[2\]a.S is 48-byte value`
	MOVQ	x_0_i+0(FP), AX // want `invalid MOVQ of x_0_i\+0\(FP\); int32 is 4-byte value`
	MOVQ	x_0_b+0(FP), AX // want `invalid offset x_0_b\+0\(FP\); expected x_0_b\+4\(FP\)`
	MOVQ	x_0_s+8(FP), AX
	MOVQ	x_0_s_base+8(FP), AX
	MOVQ	x_0_s+16(FP), AX // want `invalid offset x_0_s\+16\(FP\); expected x_0_s\+8\(FP\), x_0_s_base\+8\(FP\), or x_0_s_len\+16\(FP\)`
	MOVQ	x_0_s_len+16(FP), AX
	MOVB	foo+25(FP), AX // want `unknown variable foo; offset 25 is x_1_i\+24\(FP\)`
	MOVQ	x_1_s+32(FP), AX
	MOVQ	x_1_s_base+32(FP), AX
	MOVQ	x_1_s+40(FP), AX // want `invalid offset x_1_s\+40\(FP\); expected x_1_s\+32\(FP\), x_1_s_base\+32\(FP\), or x_1_s_len\+40\(FP\)`
	MOVQ	x_1_s_len+40(FP), AX
	RET

TEXT ·returnint(SB),0,$0-8
	MOVB	AX, ret+0(FP) // want `invalid MOVB of ret\+0\(FP\); int is 8-byte value`
	MOVW	AX, ret+0(FP) // want `invalid MOVW of ret\+0\(FP\); int is 8-byte value`
	MOVL	AX, ret+0(FP) // want `invalid MOVL of ret\+0\(FP\); int is 8-byte value`
	MOVQ	AX, ret+0(FP)
	MOVQ	AX, ret+1(FP) // want `invalid offset ret\+1\(FP\); expected ret\+0\(FP\)`
	MOVQ	AX, r+0(FP) // want `unknown variable r; offset 0 is ret\+0\(FP\)`
	RET

TEXT ·returnbyte(SB),0,$0-9
	MOVQ	x+0(FP), AX
	MOVB	AX, ret+8(FP)
	MOVW	AX, ret+8(FP) // want `invalid MOVW of ret\+8\(FP\); byte is 1-byte value`
	MOVL	AX, ret+8(FP) // want `invalid MOVL of ret\+8\(FP\); byte is 1-byte value`
	MOVQ	AX, ret+8(FP) // want `invalid MOVQ of ret\+8\(FP\); byte is 1-byte value`
	MOVB	AX, ret+7(FP) // want `invalid offset ret\+7\(FP\); expected ret\+8\(FP\)`
	RET

TEXT ·returnnamed(SB),0,$0-41
	MOVB	x+0(FP), AX
	MOVQ	AX, r1+8(FP)
	MOVW	AX, r2+16(FP)
	MOVQ	AX, r3+24(FP)
	MOVQ	AX, r3_base+24(FP)
	MOVQ	AX, r3_len+32(FP)
	MOVB	AX, r4+40(FP)
	MOVL	AX, r1+8(FP) // want `invalid MOVL of r1\+8\(FP\); int is 8-byte value`
	RET

TEXT ·returnintmissing(SB),0,$0-8
	RET // want `RET without writing to 8-byte ret\+0\(FP\)`


// issue 15271
TEXT ·f15271(SB), NOSPLIT, $0-4
    // Stick 123 into the low 32 bits of X0.
    MOVQ $123, AX
    PINSRD $0, AX, X0

    // Return them.
    PEXTRD $0, X0, x+0(FP)
    RET

// issue 17584
TEXT ·f17584(SB), NOSPLIT, $12
	MOVSS	x+0(FP), X0
	MOVSS	y_real+4(FP), X0
	MOVSS	y_imag+8(FP), X0
	RET

// issue 29318
TEXT ·f29318(SB), NOSPLIT, $32
	MOVQ	x_0_1+8(FP), AX
	MOVQ	x_1_1+24(FP), CX
	RET

// ABI selector
TEXT ·pickStableABI<ABI0>(SB), NOSPLIT, $32
	MOVQ	x+0(FP), AX
	RET

// ABI selector
TEXT ·pickInternalABI<ABIInternal>(SB), NOSPLIT, $32
	MOVQ	x+0(FP), AX
	RET

// ABI selector
TEXT ·pickFutureABI<ABISomethingNotyetInvented>(SB), NOSPLIT, $32
	MOVQ	x+0(FP), AX
	RET

// return jump
TEXT ·retjmp(SB), NOSPLIT, $0-8
	RET	retjmp1(SB) // It's okay to not write results if there's a tail call.
