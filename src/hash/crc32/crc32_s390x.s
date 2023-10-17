// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "textflag.h"

// Vector register range containing CRC-32 constants

#define CONST_PERM_LE2BE        V9
#define CONST_R2R1              V10
#define CONST_R4R3              V11
#define CONST_R5                V12
#define CONST_RU_POLY           V13
#define CONST_CRC_POLY          V14


// The CRC-32 constant block contains reduction constants to fold and
// process particular chunks of the input data stream in parallel.
//
// Note that the constant definitions below are extended in order to compute
// intermediate results with a single VECTOR GALOIS FIELD MULTIPLY instruction.
// The rightmost doubleword can be 0 to prevent contribution to the result or
// can be multiplied by 1 to perform an XOR without the need for a separate
// VECTOR EXCLUSIVE OR instruction.
//
// The polynomials used are bit-reflected:
//
//            IEEE: P'(x) = 0x0edb88320
//      Castagnoli: P'(x) = 0x082f63b78


// IEEE polynomial constants
DATA    ·crclecons+0(SB)/8,  $0x0F0E0D0C0B0A0908       // LE-to-BE mask
DATA    ·crclecons+8(SB)/8,  $0x0706050403020100
DATA    ·crclecons+16(SB)/8, $0x00000001c6e41596       // R2
DATA    ·crclecons+24(SB)/8, $0x0000000154442bd4       // R1
DATA    ·crclecons+32(SB)/8, $0x00000000ccaa009e       // R4
DATA    ·crclecons+40(SB)/8, $0x00000001751997d0       // R3
DATA    ·crclecons+48(SB)/8, $0x0000000000000000
DATA    ·crclecons+56(SB)/8, $0x0000000163cd6124       // R5
DATA    ·crclecons+64(SB)/8, $0x0000000000000000
DATA    ·crclecons+72(SB)/8, $0x00000001F7011641       // u'
DATA    ·crclecons+80(SB)/8, $0x0000000000000000
DATA    ·crclecons+88(SB)/8, $0x00000001DB710641       // P'(x) << 1

GLOBL    ·crclecons(SB),RODATA, $144

// Castagonli Polynomial constants
DATA    ·crcclecons+0(SB)/8,  $0x0F0E0D0C0B0A0908      // LE-to-BE mask
DATA    ·crcclecons+8(SB)/8,  $0x0706050403020100
DATA    ·crcclecons+16(SB)/8, $0x000000009e4addf8      // R2
DATA    ·crcclecons+24(SB)/8, $0x00000000740eef02      // R1
DATA    ·crcclecons+32(SB)/8, $0x000000014cd00bd6      // R4
DATA    ·crcclecons+40(SB)/8, $0x00000000f20c0dfe      // R3
DATA    ·crcclecons+48(SB)/8, $0x0000000000000000
DATA    ·crcclecons+56(SB)/8, $0x00000000dd45aab8      // R5
DATA    ·crcclecons+64(SB)/8, $0x0000000000000000
DATA    ·crcclecons+72(SB)/8, $0x00000000dea713f1      // u'
DATA    ·crcclecons+80(SB)/8, $0x0000000000000000
DATA    ·crcclecons+88(SB)/8, $0x0000000105ec76f0      // P'(x) << 1

GLOBL   ·crcclecons(SB),RODATA, $144

// The CRC-32 function(s) use these calling conventions:
//
// Parameters:
//
//      R2:    Initial CRC value, typically ~0; and final CRC (return) value.
//      R3:    Input buffer pointer, performance might be improved if the
//             buffer is on a doubleword boundary.
//      R4:    Length of the buffer, must be 64 bytes or greater.
//
// Register usage:
//
//      R5:     CRC-32 constant pool base pointer.
//      V0:     Initial CRC value and intermediate constants and results.
//      V1..V4: Data for CRC computation.
//      V5..V8: Next data chunks that are fetched from the input buffer.
//
//      V9..V14: CRC-32 constants.

// func vectorizedIEEE(crc uint32, p []byte) uint32
TEXT ·vectorizedIEEE(SB),NOSPLIT,$0
	MOVWZ   crc+0(FP), R2     // R2 stores the CRC value
	MOVD    p+8(FP), R3       // data pointer
	MOVD    p_len+16(FP), R4  // len(p)

	MOVD    $·crclecons(SB), R5
	BR      vectorizedBody<>(SB)

// func vectorizedCastagnoli(crc uint32, p []byte) uint32
TEXT ·vectorizedCastagnoli(SB),NOSPLIT,$0
	MOVWZ   crc+0(FP), R2     // R2 stores the CRC value
	MOVD    p+8(FP), R3       // data pointer
	MOVD    p_len+16(FP), R4  // len(p)

	// R5: crc-32 constant pool base pointer, constant is used to reduce crc
	MOVD    $·crcclecons(SB), R5
	BR      vectorizedBody<>(SB)

TEXT vectorizedBody<>(SB),NOSPLIT,$0
	XOR     $0xffffffff, R2 // NOTW R2
	VLM     0(R5), CONST_PERM_LE2BE, CONST_CRC_POLY

	// Load the initial CRC value into the rightmost word of V0
	VZERO   V0
	VLVGF   $3, R2, V0

	// Crash if the input size is less than 64-bytes.
	CMP     R4, $64
	BLT     crash

	// Load a 64-byte data chunk and XOR with CRC
	VLM     0(R3), V1, V4    // 64-bytes into V1..V4

	// Reflect the data if the CRC operation is in the bit-reflected domain
	VPERM   V1, V1, CONST_PERM_LE2BE, V1
	VPERM   V2, V2, CONST_PERM_LE2BE, V2
	VPERM   V3, V3, CONST_PERM_LE2BE, V3
	VPERM   V4, V4, CONST_PERM_LE2BE, V4

	VX      V0, V1, V1     // V1 ^= CRC
	ADD     $64, R3        // BUF = BUF + 64
	ADD     $(-64), R4

	// Check remaining buffer size and jump to proper folding method
	CMP     R4, $64
	BLT     less_than_64bytes

fold_64bytes_loop:
	// Load the next 64-byte data chunk into V5 to V8
	VLM     0(R3), V5, V8
	VPERM   V5, V5, CONST_PERM_LE2BE, V5
	VPERM   V6, V6, CONST_PERM_LE2BE, V6
	VPERM   V7, V7, CONST_PERM_LE2BE, V7
	VPERM   V8, V8, CONST_PERM_LE2BE, V8


	// Perform a GF(2) multiplication of the doublewords in V1 with
	// the reduction constants in V0.  The intermediate result is
	// then folded (accumulated) with the next data chunk in V5 and
	// stored in V1.  Repeat this step for the register contents
	// in V2, V3, and V4 respectively.

	VGFMAG  CONST_R2R1, V1, V5, V1
	VGFMAG  CONST_R2R1, V2, V6, V2
	VGFMAG  CONST_R2R1, V3, V7, V3
	VGFMAG  CONST_R2R1, V4, V8 ,V4

	// Adjust buffer pointer and length for next loop
	ADD     $64, R3                  // BUF = BUF + 64
	ADD     $(-64), R4               // LEN = LEN - 64

	CMP     R4, $64
	BGE     fold_64bytes_loop

less_than_64bytes:
	// Fold V1 to V4 into a single 128-bit value in V1
	VGFMAG  CONST_R4R3, V1, V2, V1
	VGFMAG  CONST_R4R3, V1, V3, V1
	VGFMAG  CONST_R4R3, V1, V4, V1

	// Check whether to continue with 64-bit folding
	CMP R4, $16
	BLT final_fold

fold_16bytes_loop:
	VL      0(R3), V2               // Load next data chunk
	VPERM   V2, V2, CONST_PERM_LE2BE, V2

	VGFMAG  CONST_R4R3, V1, V2, V1  // Fold next data chunk

	// Adjust buffer pointer and size for folding next data chunk
	ADD     $16, R3
	ADD     $-16, R4

	// Process remaining data chunks
	CMP     R4 ,$16
	BGE     fold_16bytes_loop

final_fold:
	VLEIB   $7, $0x40, V9
	VSRLB   V9, CONST_R4R3, V0
	VLEIG   $0, $1, V0

	VGFMG   V0, V1, V1

	VLEIB   $7, $0x20, V9         // Shift by words
	VSRLB   V9, V1, V2            // Store remaining bits in V2
	VUPLLF  V1, V1                // Split rightmost doubleword
	VGFMAG  CONST_R5, V1, V2, V1  // V1 = (V1 * R5) XOR V2


	// The input values to the Barret reduction are the degree-63 polynomial
	// in V1 (R(x)), degree-32 generator polynomial, and the reduction
	// constant u.  The Barret reduction result is the CRC value of R(x) mod
	// P(x).
	//
	// The Barret reduction algorithm is defined as:
	//
	//    1. T1(x) = floor( R(x) / x^32 ) GF2MUL u
	//    2. T2(x) = floor( T1(x) / x^32 ) GF2MUL P(x)
	//    3. C(x)  = R(x) XOR T2(x) mod x^32
	//
	// Note: To compensate the division by x^32, use the vector unpack
	// instruction to move the leftmost word into the leftmost doubleword
	// of the vector register.  The rightmost doubleword is multiplied
	// with zero to not contribute to the intermediate results.


	// T1(x) = floor( R(x) / x^32 ) GF2MUL u
	VUPLLF  V1, V2
	VGFMG   CONST_RU_POLY, V2, V2


	// Compute the GF(2) product of the CRC polynomial in VO with T1(x) in
	// V2 and XOR the intermediate result, T2(x),  with the value in V1.
	// The final result is in the rightmost word of V2.

	VUPLLF  V2, V2
	VGFMAG  CONST_CRC_POLY, V2, V1, V2

done:
	VLGVF   $2, V2, R2
	XOR     $0xffffffff, R2 // NOTW R2
	MOVWZ   R2, ret + 32(FP)
	RET

crash:
	MOVD    $0, (R0) // input size is less than 64-bytes
