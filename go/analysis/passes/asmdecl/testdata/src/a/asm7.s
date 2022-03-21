// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

TEXT ·arg1(SB),0,$0-2
	MOVB	x+0(FP), R3
	MOVBZ	y+1(FP), R4
	MOVH	x+0(FP), R3 // want `\[(ppc64|ppc64le)\] arg1: invalid MOVH of x\+0\(FP\); int8 is 1-byte value`
	MOVHZ	y+1(FP), R3 // want `invalid MOVHZ of y\+1\(FP\); uint8 is 1-byte value`
	MOVW	x+0(FP), R3 // want `invalid MOVW of x\+0\(FP\); int8 is 1-byte value`
	MOVWZ	y+1(FP), R3 // want `invalid MOVWZ of y\+1\(FP\); uint8 is 1-byte value`
	MOVD	x+0(FP), R3 // want `invalid MOVD of x\+0\(FP\); int8 is 1-byte value`
	MOVD	y+1(FP), R3 // want `invalid MOVD of y\+1\(FP\); uint8 is 1-byte value`
	MOVB	x+1(FP), R3 // want `invalid offset x\+1\(FP\); expected x\+0\(FP\)`
	MOVBZ	y+2(FP), R3 // want `invalid offset y\+2\(FP\); expected y\+1\(FP\)`
	MOVB	16(R1), R3 // want `16\(R1\) should be x\+0\(FP\)`
	MOVB	17(R1), R3 // want `17\(R1\) should be y\+1\(FP\)`
	MOVB	18(R1), R3 // want `use of 18\(R1\) points beyond argument frame`
	RET

TEXT ·arg2(SB),0,$0-4
	MOVBZ	x+0(FP), R3 // want `arg2: invalid MOVBZ of x\+0\(FP\); int16 is 2-byte value`
	MOVB	y+2(FP), R3 // want `invalid MOVB of y\+2\(FP\); uint16 is 2-byte value`
	MOVHZ	x+0(FP), R3
	MOVH	y+2(FP), R4
	MOVWZ	x+0(FP), R3 // want `invalid MOVWZ of x\+0\(FP\); int16 is 2-byte value`
	MOVW	y+2(FP), R3 // want `invalid MOVW of y\+2\(FP\); uint16 is 2-byte value`
	MOVD	x+0(FP), R3 // want `invalid MOVD of x\+0\(FP\); int16 is 2-byte value`
	MOVD	y+2(FP), R3 // want `invalid MOVD of y\+2\(FP\); uint16 is 2-byte value`
	MOVHZ	x+2(FP), R3 // want `invalid offset x\+2\(FP\); expected x\+0\(FP\)`
	MOVH	y+0(FP), R3 // want `invalid offset y\+0\(FP\); expected y\+2\(FP\)`
	RET

TEXT ·arg4(SB),0,$0-2 // want `arg4: wrong argument size 2; expected \$\.\.\.-8`
	MOVB	x+0(FP), R3 // want `invalid MOVB of x\+0\(FP\); int32 is 4-byte value`
	MOVB	y+4(FP), R4 // want `invalid MOVB of y\+4\(FP\); uint32 is 4-byte value`
	MOVH	x+0(FP), R3 // want `invalid MOVH of x\+0\(FP\); int32 is 4-byte value`
	MOVH	y+4(FP), R3 // want `invalid MOVH of y\+4\(FP\); uint32 is 4-byte value`
	MOVW	x+0(FP), R3
	MOVW	y+4(FP), R3
	MOVD	x+0(FP), R3 // want `invalid MOVD of x\+0\(FP\); int32 is 4-byte value`
	MOVD	y+4(FP), R3 // want `invalid MOVD of y\+4\(FP\); uint32 is 4-byte value`
	MOVW	x+4(FP), R3 // want `invalid offset x\+4\(FP\); expected x\+0\(FP\)`
	MOVW	y+2(FP), R3 // want `invalid offset y\+2\(FP\); expected y\+4\(FP\)`
	RET

TEXT ·arg8(SB),7,$0-2 // want `wrong argument size 2; expected \$\.\.\.-16`
	MOVB	x+0(FP), R3 // want `invalid MOVB of x\+0\(FP\); int64 is 8-byte value`
	MOVB	y+8(FP), R4 // want `invalid MOVB of y\+8\(FP\); uint64 is 8-byte value`
	MOVH	x+0(FP), R3 // want `invalid MOVH of x\+0\(FP\); int64 is 8-byte value`
	MOVH	y+8(FP), R3 // want `invalid MOVH of y\+8\(FP\); uint64 is 8-byte value`
	MOVW	x+0(FP), R3 // want `invalid MOVW of x\+0\(FP\); int64 is 8-byte value`
	MOVW	y+8(FP), R3 // want `invalid MOVW of y\+8\(FP\); uint64 is 8-byte value`
	MOVD	x+0(FP), R3
	MOVD	y+8(FP), R3
	MOVD	x+8(FP), R3 // want `invalid offset x\+8\(FP\); expected x\+0\(FP\)`
	MOVD	y+2(FP), R3 // want `invalid offset y\+2\(FP\); expected y\+8\(FP\)`
	RET

TEXT ·argint(SB),0,$0-2 // want `wrong argument size 2; expected \$\.\.\.-16`
	MOVB	x+0(FP), R3 // want `invalid MOVB of x\+0\(FP\); int is 8-byte value`
	MOVB	y+8(FP), R4 // want `invalid MOVB of y\+8\(FP\); uint is 8-byte value`
	MOVH	x+0(FP), R3 // want `invalid MOVH of x\+0\(FP\); int is 8-byte value`
	MOVH	y+8(FP), R3 // want `invalid MOVH of y\+8\(FP\); uint is 8-byte value`
	MOVW	x+0(FP), R3 // want `invalid MOVW of x\+0\(FP\); int is 8-byte value`
	MOVW	y+8(FP), R3 // want `invalid MOVW of y\+8\(FP\); uint is 8-byte value`
	MOVD	x+0(FP), R3
	MOVD	y+8(FP), R3
	MOVD	x+8(FP), R3 // want `invalid offset x\+8\(FP\); expected x\+0\(FP\)`
	MOVD	y+2(FP), R3 // want `invalid offset y\+2\(FP\); expected y\+8\(FP\)`
	RET

TEXT ·argptr(SB),7,$0-2 // want `wrong argument size 2; expected \$\.\.\.-40`
	MOVB	x+0(FP), R3 // want `invalid MOVB of x\+0\(FP\); \*byte is 8-byte value`
	MOVB	y+8(FP), R4 // want `invalid MOVB of y\+8\(FP\); \*byte is 8-byte value`
	MOVH	x+0(FP), R3 // want `invalid MOVH of x\+0\(FP\); \*byte is 8-byte value`
	MOVH	y+8(FP), R3 // want `invalid MOVH of y\+8\(FP\); \*byte is 8-byte value`
	MOVW	x+0(FP), R3 // want `invalid MOVW of x\+0\(FP\); \*byte is 8-byte value`
	MOVW	y+8(FP), R3 // want `invalid MOVW of y\+8\(FP\); \*byte is 8-byte value`
	MOVD	x+0(FP), R3
	MOVD	y+8(FP), R3
	MOVD	x+8(FP), R3 // want `invalid offset x\+8\(FP\); expected x\+0\(FP\)`
	MOVD	y+2(FP), R3 // want `invalid offset y\+2\(FP\); expected y\+8\(FP\)`
	MOVW	c+16(FP), R3 // want `invalid MOVW of c\+16\(FP\); chan int is 8-byte value`
	MOVW	m+24(FP), R3 // want `invalid MOVW of m\+24\(FP\); map\[int\]int is 8-byte value`
	MOVW	f+32(FP), R3 // want `invalid MOVW of f\+32\(FP\); func\(\) is 8-byte value`
	RET

TEXT ·argstring(SB),0,$32 // want `wrong argument size 0; expected \$\.\.\.-32`
	MOVH	x+0(FP), R3 // want `invalid MOVH of x\+0\(FP\); string base is 8-byte value`
	MOVW	x+0(FP), R3 // want `invalid MOVW of x\+0\(FP\); string base is 8-byte value`
	MOVD	x+0(FP), R3
	MOVH	x_base+0(FP), R3 // want `invalid MOVH of x_base\+0\(FP\); string base is 8-byte value`
	MOVW	x_base+0(FP), R3 // want `invalid MOVW of x_base\+0\(FP\); string base is 8-byte value`
	MOVD	x_base+0(FP), R3
	MOVH	x_len+0(FP), R3 // want `invalid offset x_len\+0\(FP\); expected x_len\+8\(FP\)`
	MOVW	x_len+0(FP), R3 // want `invalid offset x_len\+0\(FP\); expected x_len\+8\(FP\)`
	MOVD	x_len+0(FP), R3 // want `invalid offset x_len\+0\(FP\); expected x_len\+8\(FP\)`
	MOVH	x_len+8(FP), R3 // want `invalid MOVH of x_len\+8\(FP\); string len is 8-byte value`
	MOVW	x_len+8(FP), R3 // want `invalid MOVW of x_len\+8\(FP\); string len is 8-byte value`
	MOVD	x_len+8(FP), R3
	MOVD	y+0(FP), R3 // want `invalid offset y\+0\(FP\); expected y\+16\(FP\)`
	MOVD	y_len+8(FP), R3 // want `invalid offset y_len\+8\(FP\); expected y_len\+24\(FP\)`
	RET

TEXT ·argslice(SB),0,$48 // want `wrong argument size 0; expected \$\.\.\.-48`
	MOVH	x+0(FP), R3 // want `invalid MOVH of x\+0\(FP\); slice base is 8-byte value`
	MOVW	x+0(FP), R3 // want `invalid MOVW of x\+0\(FP\); slice base is 8-byte value`
	MOVD	x+0(FP), R3
	MOVH	x_base+0(FP), R3 // want `invalid MOVH of x_base\+0\(FP\); slice base is 8-byte value`
	MOVW	x_base+0(FP), R3 // want `invalid MOVW of x_base\+0\(FP\); slice base is 8-byte value`
	MOVD	x_base+0(FP), R3
	MOVH	x_len+0(FP), R3 // want `invalid offset x_len\+0\(FP\); expected x_len\+8\(FP\)`
	MOVW	x_len+0(FP), R3 // want `invalid offset x_len\+0\(FP\); expected x_len\+8\(FP\)`
	MOVD	x_len+0(FP), R3 // want `invalid offset x_len\+0\(FP\); expected x_len\+8\(FP\)`
	MOVH	x_len+8(FP), R3 // want `invalid MOVH of x_len\+8\(FP\); slice len is 8-byte value`
	MOVW	x_len+8(FP), R3 // want `invalid MOVW of x_len\+8\(FP\); slice len is 8-byte value`
	MOVD	x_len+8(FP), R3
	MOVH	x_cap+0(FP), R3 // want `invalid offset x_cap\+0\(FP\); expected x_cap\+16\(FP\)`
	MOVW	x_cap+0(FP), R3 // want `invalid offset x_cap\+0\(FP\); expected x_cap\+16\(FP\)`
	MOVD	x_cap+0(FP), R3 // want `invalid offset x_cap\+0\(FP\); expected x_cap\+16\(FP\)`
	MOVH	x_cap+16(FP), R3 // want `invalid MOVH of x_cap\+16\(FP\); slice cap is 8-byte value`
	MOVW	x_cap+16(FP), R3 // want `invalid MOVW of x_cap\+16\(FP\); slice cap is 8-byte value`
	MOVD	x_cap+16(FP), R3
	MOVD	y+0(FP), R3 // want `invalid offset y\+0\(FP\); expected y\+24\(FP\)`
	MOVD	y_len+8(FP), R3 // want `invalid offset y_len\+8\(FP\); expected y_len\+32\(FP\)`
	MOVD	y_cap+16(FP), R3 // want `invalid offset y_cap\+16\(FP\); expected y_cap\+40\(FP\)`
	RET

TEXT ·argiface(SB),0,$0-32
	MOVH	x+0(FP), R3 // want `invalid MOVH of x\+0\(FP\); interface type is 8-byte value`
	MOVW	x+0(FP), R3 // want `invalid MOVW of x\+0\(FP\); interface type is 8-byte value`
	MOVD	x+0(FP), R3
	MOVH	x_type+0(FP), R3 // want `invalid MOVH of x_type\+0\(FP\); interface type is 8-byte value`
	MOVW	x_type+0(FP), R3 // want `invalid MOVW of x_type\+0\(FP\); interface type is 8-byte value`
	MOVD	x_type+0(FP), R3
	MOVD	x_itable+0(FP), R3 // want `unknown variable x_itable; offset 0 is x_type\+0\(FP\)`
	MOVD	x_itable+1(FP), R3 // want `unknown variable x_itable; offset 1 is x_type\+0\(FP\)`
	MOVH	x_data+0(FP), R3 // want `invalid offset x_data\+0\(FP\); expected x_data\+8\(FP\)`
	MOVW	x_data+0(FP), R3 // want `invalid offset x_data\+0\(FP\); expected x_data\+8\(FP\)`
	MOVD	x_data+0(FP), R3 // want `invalid offset x_data\+0\(FP\); expected x_data\+8\(FP\)`
	MOVH	x_data+8(FP), R3 // want `invalid MOVH of x_data\+8\(FP\); interface data is 8-byte value`
	MOVW	x_data+8(FP), R3 // want `invalid MOVW of x_data\+8\(FP\); interface data is 8-byte value`
	MOVD	x_data+8(FP), R3
	MOVH	y+16(FP), R3 // want `invalid MOVH of y\+16\(FP\); interface itable is 8-byte value`
	MOVW	y+16(FP), R3 // want `invalid MOVW of y\+16\(FP\); interface itable is 8-byte value`
	MOVD	y+16(FP), R3
	MOVH	y_itable+16(FP), R3 // want `invalid MOVH of y_itable\+16\(FP\); interface itable is 8-byte value`
	MOVW	y_itable+16(FP), R3 // want `invalid MOVW of y_itable\+16\(FP\); interface itable is 8-byte value`
	MOVD	y_itable+16(FP), R3
	MOVD	y_type+16(FP), R3 // want `unknown variable y_type; offset 16 is y_itable\+16\(FP\)`
	MOVH	y_data+16(FP), R3 // want `invalid offset y_data\+16\(FP\); expected y_data\+24\(FP\)`
	MOVW	y_data+16(FP), R3 // want `invalid offset y_data\+16\(FP\); expected y_data\+24\(FP\)`
	MOVD	y_data+16(FP), R3 // want `invalid offset y_data\+16\(FP\); expected y_data\+24\(FP\)`
	MOVH	y_data+24(FP), R3 // want `invalid MOVH of y_data\+24\(FP\); interface data is 8-byte value`
	MOVW	y_data+24(FP), R3 // want `invalid MOVW of y_data\+24\(FP\); interface data is 8-byte value`
	MOVD	y_data+24(FP), R3
	RET

TEXT ·returnint(SB),0,$0-8
	MOVB	R3, ret+0(FP) // want `invalid MOVB of ret\+0\(FP\); int is 8-byte value`
	MOVH	R3, ret+0(FP) // want `invalid MOVH of ret\+0\(FP\); int is 8-byte value`
	MOVW	R3, ret+0(FP) // want `invalid MOVW of ret\+0\(FP\); int is 8-byte value`
	MOVD	R3, ret+0(FP)
	MOVD	R3, ret+1(FP) // want `invalid offset ret\+1\(FP\); expected ret\+0\(FP\)`
	MOVD	R3, r+0(FP) // want `unknown variable r; offset 0 is ret\+0\(FP\)`
	RET

TEXT ·returnbyte(SB),0,$0-9
	MOVD	x+0(FP), R3
	MOVB	R3, ret+8(FP)
	MOVH	R3, ret+8(FP) // want `invalid MOVH of ret\+8\(FP\); byte is 1-byte value`
	MOVW	R3, ret+8(FP) // want `invalid MOVW of ret\+8\(FP\); byte is 1-byte value`
	MOVD	R3, ret+8(FP) // want `invalid MOVD of ret\+8\(FP\); byte is 1-byte value`
	MOVB	R3, ret+7(FP) // want `invalid offset ret\+7\(FP\); expected ret\+8\(FP\)`
	RET

TEXT ·returnnamed(SB),0,$0-41
	MOVB	x+0(FP), R3
	MOVD	R3, r1+8(FP)
	MOVH	R3, r2+16(FP)
	MOVD	R3, r3+24(FP)
	MOVD	R3, r3_base+24(FP)
	MOVD	R3, r3_len+32(FP)
	MOVB	R3, r4+40(FP)
	MOVW	R3, r1+8(FP) // want `invalid MOVW of r1\+8\(FP\); int is 8-byte value`
	RET

TEXT ·returnintmissing(SB),0,$0-8
	RET // want `RET without writing to 8-byte ret\+0\(FP\)`

// writing to result in ABIInternal function
TEXT ·returnABIInternal<ABIInternal>(SB), NOSPLIT, $8
	MOVD	$123, R3
	RET
TEXT ·returnmissingABIInternal<ABIInternal>(SB), NOSPLIT, $8
	MOVD	$123, R10
	RET // want `RET without writing to result register`
