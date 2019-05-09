// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mipsle

TEXT ·arg1(SB),0,$0-2
	MOVB	x+0(FP), R1
	MOVBU	y+1(FP), R2
	MOVH	x+0(FP), R1 // want `\[mipsle\] arg1: invalid MOVH of x\+0\(FP\); int8 is 1-byte value`
	MOVHU	y+1(FP), R1 // want `invalid MOVHU of y\+1\(FP\); uint8 is 1-byte value`
	MOVW	x+0(FP), R1 // want `invalid MOVW of x\+0\(FP\); int8 is 1-byte value`
	MOVWU	y+1(FP), R1 // want `invalid MOVWU of y\+1\(FP\); uint8 is 1-byte value`
	MOVW	y+1(FP), R1 // want `invalid MOVW of y\+1\(FP\); uint8 is 1-byte value`
	MOVB	x+1(FP), R1 // want `invalid offset x\+1\(FP\); expected x\+0\(FP\)`
	MOVBU	y+2(FP), R1 // want `invalid offset y\+2\(FP\); expected y\+1\(FP\)`
	MOVB	8(R29), R1 // want `8\(R29\) should be x\+0\(FP\)`
	MOVB	9(R29), R1 // want `9\(R29\) should be y\+1\(FP\)`
	MOVB	10(R29), R1 // want `use of 10\(R29\) points beyond argument frame`
	RET

TEXT ·arg2(SB),0,$0-4
	MOVBU	x+0(FP), R1 // want `arg2: invalid MOVBU of x\+0\(FP\); int16 is 2-byte value`
	MOVB	y+2(FP), R1 // want `invalid MOVB of y\+2\(FP\); uint16 is 2-byte value`
	MOVHU	x+0(FP), R1
	MOVH	y+2(FP), R2
	MOVWU	x+0(FP), R1 // want `invalid MOVWU of x\+0\(FP\); int16 is 2-byte value`
	MOVW	y+2(FP), R1 // want `invalid MOVW of y\+2\(FP\); uint16 is 2-byte value`
	MOVHU	x+2(FP), R1 // want `invalid offset x\+2\(FP\); expected x\+0\(FP\)`
	MOVH	y+0(FP), R1 // want `invalid offset y\+0\(FP\); expected y\+2\(FP\)`
	RET

TEXT ·arg4(SB),0,$0-2 // want `arg4: wrong argument size 2; expected \$\.\.\.-8`
	MOVB	x+0(FP), R1 // want `invalid MOVB of x\+0\(FP\); int32 is 4-byte value`
	MOVB	y+4(FP), R2 // want `invalid MOVB of y\+4\(FP\); uint32 is 4-byte value`
	MOVH	x+0(FP), R1 // want `invalid MOVH of x\+0\(FP\); int32 is 4-byte value`
	MOVH	y+4(FP), R1 // want `invalid MOVH of y\+4\(FP\); uint32 is 4-byte value`
	MOVW	x+0(FP), R1
	MOVW	y+4(FP), R1
	MOVW	x+4(FP), R1 // want `invalid offset x\+4\(FP\); expected x\+0\(FP\)`
	MOVW	y+2(FP), R1 // want `invalid offset y\+2\(FP\); expected y\+4\(FP\)`
	RET

TEXT ·arg8(SB),7,$0-2 // want `wrong argument size 2; expected \$\.\.\.-16`
	MOVB	x+0(FP), R1 // want `invalid MOVB of x\+0\(FP\); int64 is 8-byte value`
	MOVB	y+8(FP), R2 // want `invalid MOVB of y\+8\(FP\); uint64 is 8-byte value`
	MOVH	x+0(FP), R1 // want `invalid MOVH of x\+0\(FP\); int64 is 8-byte value`
	MOVH	y+8(FP), R1 // want `invalid MOVH of y\+8\(FP\); uint64 is 8-byte value`
	MOVW	x+0(FP), R1 // want `invalid MOVW of x\+0\(FP\); int64 is 8-byte value containing x_lo\+0\(FP\) and x_hi\+4\(FP\)`
	MOVW	$x+0(FP), R1 // ok
	MOVW	x_lo+0(FP), R1
	MOVW	x_hi+4(FP), R1
	MOVW	y+8(FP), R1 // want `invalid MOVW of y\+8\(FP\); uint64 is 8-byte value containing y_lo\+8\(FP\) and y_hi\+12\(FP\)`
	MOVW	y_lo+8(FP),  R1
	MOVW	y_hi+12(FP), R1
	RET

TEXT ·argint(SB),0,$0-2 // want `wrong argument size 2; expected \$\.\.\.-8`
	MOVB	x+0(FP), R1 // want `invalid MOVB of x\+0\(FP\); int is 4-byte value`
	MOVB	y+4(FP), R2 // want `invalid MOVB of y\+4\(FP\); uint is 4-byte value`
	MOVH	x+0(FP), R1 // want `invalid MOVH of x\+0\(FP\); int is 4-byte value`
	MOVH	y+4(FP), R1 // want `invalid MOVH of y\+4\(FP\); uint is 4-byte value`
	MOVW	x+0(FP), R1
	MOVW	y+4(FP), R1
	MOVW	x+4(FP), R1 // want `invalid offset x\+4\(FP\); expected x\+0\(FP\)`
	MOVW	y+2(FP), R1 // want `invalid offset y\+2\(FP\); expected y\+4\(FP\)`
	RET

TEXT ·argptr(SB),7,$0-2 // want `wrong argument size 2; expected \$\.\.\.-20`
	MOVB	x+0(FP), R1 // want `invalid MOVB of x\+0\(FP\); \*byte is 4-byte value`
	MOVB	y+4(FP), R2 // want `invalid MOVB of y\+4\(FP\); \*byte is 4-byte value`
	MOVH	x+0(FP), R1 // want `invalid MOVH of x\+0\(FP\); \*byte is 4-byte value`
	MOVH	y+4(FP), R1 // want `invalid MOVH of y\+4\(FP\); \*byte is 4-byte value`
	MOVW	x+0(FP), R1
	MOVW	y+4(FP), R1
	MOVW	x+4(FP), R1 // want `invalid offset x\+4\(FP\); expected x\+0\(FP\)`
	MOVW	y+2(FP), R1 // want `invalid offset y\+2\(FP\); expected y\+4\(FP\)`
	MOVH	c+8(FP), R1 // want `invalid MOVH of c\+8\(FP\); chan int is 4-byte value`
	MOVH	m+12(FP), R1 // want `invalid MOVH of m\+12\(FP\); map\[int\]int is 4-byte value`
	MOVH	f+16(FP), R1 // want `invalid MOVH of f\+16\(FP\); func\(\) is 4-byte value`
	RET

TEXT ·argstring(SB),0,$16 // want `wrong argument size 0; expected \$\.\.\.-16`
	MOVH	x+0(FP), R1 // want `invalid MOVH of x\+0\(FP\); string base is 4-byte value`
	MOVW	x+0(FP), R1
	MOVH	x_base+0(FP), R1 // want `invalid MOVH of x_base\+0\(FP\); string base is 4-byte value`
	MOVW	x_base+0(FP), R1
	MOVH	x_len+0(FP), R1 // want `invalid offset x_len\+0\(FP\); expected x_len\+4\(FP\)`
	MOVW	x_len+0(FP), R1 // want `invalid offset x_len\+0\(FP\); expected x_len\+4\(FP\)`
	MOVH	x_len+4(FP), R1 // want `invalid MOVH of x_len\+4\(FP\); string len is 4-byte value`
	MOVW	x_len+4(FP), R1
	MOVW	y+0(FP), R1 // want `invalid offset y\+0\(FP\); expected y\+8\(FP\)`
	MOVW	y_len+4(FP), R1 // want `invalid offset y_len\+4\(FP\); expected y_len\+12\(FP\)`
	RET

TEXT ·argslice(SB),0,$24 // want `wrong argument size 0; expected \$\.\.\.-24`
	MOVH	x+0(FP), R1 // want `invalid MOVH of x\+0\(FP\); slice base is 4-byte value`
	MOVW	x+0(FP), R1
	MOVH	x_base+0(FP), R1 // want `invalid MOVH of x_base\+0\(FP\); slice base is 4-byte value`
	MOVW	x_base+0(FP), R1
	MOVH	x_len+0(FP), R1 // want `invalid offset x_len\+0\(FP\); expected x_len\+4\(FP\)`
	MOVW	x_len+0(FP), R1 // want `invalid offset x_len\+0\(FP\); expected x_len\+4\(FP\)`
	MOVH	x_len+4(FP), R1 // want `invalid MOVH of x_len\+4\(FP\); slice len is 4-byte value`
	MOVW	x_len+4(FP), R1
	MOVH	x_cap+0(FP), R1 // want `invalid offset x_cap\+0\(FP\); expected x_cap\+8\(FP\)`
	MOVW	x_cap+0(FP), R1 // want `invalid offset x_cap\+0\(FP\); expected x_cap\+8\(FP\)`
	MOVH	x_cap+8(FP), R1 // want `invalid MOVH of x_cap\+8\(FP\); slice cap is 4-byte value`
	MOVW	x_cap+8(FP), R1
	MOVW	y+0(FP), R1 // want `invalid offset y\+0\(FP\); expected y\+12\(FP\)`
	MOVW	y_len+4(FP), R1 // want `invalid offset y_len\+4\(FP\); expected y_len\+16\(FP\)`
	MOVW	y_cap+8(FP), R1 // want `invalid offset y_cap\+8\(FP\); expected y_cap\+20\(FP\)`
	RET

TEXT ·argiface(SB),0,$0-16
	MOVH	x+0(FP), R1 // want `invalid MOVH of x\+0\(FP\); interface type is 4-byte value`
	MOVW	x+0(FP), R1
	MOVH	x_type+0(FP), R1 // want `invalid MOVH of x_type\+0\(FP\); interface type is 4-byte value`
	MOVW	x_type+0(FP), R1
	MOVQ	x_itable+0(FP), R1 // want `unknown variable x_itable; offset 0 is x_type\+0\(FP\)`
	MOVQ	x_itable+1(FP), R1 // want `unknown variable x_itable; offset 1 is x_type\+0\(FP\)`
	MOVH	x_data+0(FP), R1 // want `invalid offset x_data\+0\(FP\); expected x_data\+4\(FP\)`
	MOVW	x_data+0(FP), R1 // want `invalid offset x_data\+0\(FP\); expected x_data\+4\(FP\)`
	MOVQ	x_data+0(FP), R1 // want `invalid offset x_data\+0\(FP\); expected x_data\+4\(FP\)`
	MOVH	x_data+4(FP), R1 // want `invalid MOVH of x_data\+4\(FP\); interface data is 4-byte value`
	MOVW	x_data+4(FP), R1
	MOVH	y+8(FP), R1 // want `invalid MOVH of y\+8\(FP\); interface itable is 4-byte value`
	MOVW	y+8(FP), R1
	MOVH	y_itable+8(FP), R1 // want `invalid MOVH of y_itable\+8\(FP\); interface itable is 4-byte value`
	MOVW	y_itable+8(FP), R1
	MOVW	y_type+8(FP), AX // want `unknown variable y_type; offset 8 is y_itable\+8\(FP\)`
	MOVH	y_data+8(FP), AX // want `invalid offset y_data\+8\(FP\); expected y_data\+12\(FP\)`
	MOVW	y_data+8(FP), AX // want `invalid offset y_data\+8\(FP\); expected y_data\+12\(FP\)`
	MOVH	y_data+12(FP), AX // want `invalid MOVH of y_data\+12\(FP\); interface data is 4-byte value`
	MOVW	y_data+12(FP), AX
	RET

TEXT ·returnbyte(SB),0,$0-5
	MOVW	x+0(FP), R1
	MOVB	R1, ret+4(FP)
	MOVH	R1, ret+4(FP) // want `invalid MOVH of ret\+4\(FP\); byte is 1-byte value`
	MOVW	R1, ret+4(FP) // want `invalid MOVW of ret\+4\(FP\); byte is 1-byte value`
	MOVB	R1, ret+3(FP) // want `invalid offset ret\+3\(FP\); expected ret\+4\(FP\)`
	RET

TEXT ·returnbyte(SB),0,$0-5
	MOVW	x+0(FP), R1
	MOVB	R1, ret+4(FP)
	MOVH	R1, ret+4(FP) // want `invalid MOVH of ret\+4\(FP\); byte is 1-byte value`
	MOVW	R1, ret+4(FP) // want `invalid MOVW of ret\+4\(FP\); byte is 1-byte value`
	MOVB	R1, ret+3(FP) // want `invalid offset ret\+3\(FP\); expected ret\+4\(FP\)`
	RET

TEXT ·returnnamed(SB),0,$0-21
	MOVB	x+0(FP), AX
	MOVW	R1, r1+4(FP)
	MOVH	R1, r2+8(FP)
	MOVW	R1, r3+12(FP)
	MOVW	R1, r3_base+12(FP)
	MOVW	R1, r3_len+16(FP)
	MOVB	R1, r4+20(FP)
	MOVB	R1, r1+4(FP) // want `invalid MOVB of r1\+4\(FP\); int is 4-byte value`
	RET

TEXT ·returnintmissing(SB),0,$0-4
	RET // want `RET without writing to 4-byte ret\+0\(FP\)`
