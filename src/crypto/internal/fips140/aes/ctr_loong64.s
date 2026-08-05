// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build !purego

// AES-CTR keystream generation for loong64 using LSX vector instructions.
//
// This file implements ctrBlocks1Asm/ctrBlocks2Asm/ctrBlocks4Asm/ctrBlocks8Asm,
// which encrypt 1/2/4/8 counter blocks per call respectively and XOR the
// resulting keystream into the source buffer.
//
// SUBBYTES and MIXCOLUMNS are macro-for-macro identical to the versions
// defined in aes_loong64.s (same VSHUFB-based S-box split and same
// XTIME-based column-mixing identity), but are redefined independently
// in this file rather than shared via a common #include. This mirrors
// the amd64/arm64 convention of aes_{amd64,arm64}.s and ctr_{amd64,arm64}.s
// not sharing macros across files: each file is self-contained and can be
// read/reviewed/regenerated without cross-file macro-expansion context.
// The only symbol actually shared across the two files is the sbox0_0
// S-box table and the shiftRows/byteSwap32/rot1/rot2/rot3 permutation
// tables, which are defined once in aes_loong64.s as package-global
// (non-<>) symbols and referenced here by plain name.

#include "textflag.h"

#define SUBBYTES(src, sbox_ptr, out) \
    VANDB   $0xe0, src, V25;     \  // V25 = src & 0xe0, keep only bit7/bit6/bit5
    VSRAB   $7,  V25,  V31;      \  // V31 = byte-wise broadcast of bit7 (0xFF/0x00)
    VSLLB   $1,  V25,  V24;      \  // shift bit6 into the sign position
    VSRAB   $7,  V24,  V30;      \  // V30 = byte-wise broadcast of bit6
    VSLLB   $2,  V25,  V24;      \  // shift bit5 into the sign position
    VSRAB   $7,  V24,  V29;      \  // V29 = byte-wise broadcast of bit5
    VANDB   $0x1f, src, V26;     \  // V26 = src & 0x1f, index for VSHUFB 32-way lookup
    \
    /* ---- 2. Chunk0/1 -> r01 (V16..V19,V28,V27 freed after use) ---- */ \
    VMOVQ   (sbox_ptr),   V16;   \
    VMOVQ   16(sbox_ptr), V17;   \
    VMOVQ   32(sbox_ptr), V18;   \
    VMOVQ   48(sbox_ptr), V19;   \
    VSHUFB  V26, V16, V17, V28;  \  // V28 = chunk0_raw
    VSHUFB  V26, V18, V19, V27;  \  // V27 = chunk1_raw
    VBITSELV V29, V27, V28, V20; \  // V20 = r01 = M5 ? chunk1 : chunk0
    \
    /* ---- 3. Chunk2/3 -> r23 ---- */ \
    VMOVQ   64(sbox_ptr),  V16;  \
    VMOVQ   80(sbox_ptr),  V17;  \
    VMOVQ   96(sbox_ptr),  V18;  \
    VMOVQ   112(sbox_ptr), V19;  \
    VSHUFB  V26, V16, V17, V28;  \  // V28 = chunk2_raw
    VSHUFB  V26, V18, V19, V27;  \  // V27 = chunk3_raw
    VBITSELV V29, V27, V28, V21; \  // V21 = r23
    \
    /* ---- 4. Merge first group (chunk0..3) -> r0123 ---- */ \
    VBITSELV V30, V21, V20, V22; \  // V22 = r0123 = M6 ? r23 : r01
    \
    /* ---- 5. Chunk4/5 -> r45 ---- */ \
    VMOVQ   128(sbox_ptr), V16;  \
    VMOVQ   144(sbox_ptr), V17;  \
    VMOVQ   160(sbox_ptr), V18;  \
    VMOVQ   176(sbox_ptr), V19;  \
    VSHUFB  V26, V16, V17, V28;  \  // V28 = chunk4_raw
    VSHUFB  V26, V18, V19, V27;  \  // V27 = chunk5_raw
    VBITSELV V29, V27, V28, V20; \  // V20 reused = r45
    \
    /* ---- 6. Chunk6/7 -> r67 ---- */ \
    VMOVQ   192(sbox_ptr), V16;  \
    VMOVQ   208(sbox_ptr), V17;  \
    VMOVQ   224(sbox_ptr), V18;  \
    VMOVQ   240(sbox_ptr), V19;  \
    VSHUFB  V26, V16, V17, V28;  \  // V28 = chunk6_raw
    VSHUFB  V26, V18, V19, V27;  \  // V27 = chunk7_raw
    VBITSELV V29, V27, V28, V21; \  // V21 reused = r67
    \
    /* ---- 7. Merge second group (chunk4..7) -> r4567 ---- */ \
    VBITSELV V30, V21, V20, V23; \  // V23 = r4567 = M6 ? r67 : r45
    \
    /* ---- 8. Final top-level merge ---- */ \
    VBITSELV V31, V23, V22, out      // out = M7 ? r4567 : r0123

// -----------------------------------------------------------------------
// MixColumns helper: multiply each byte of Vn by 2 in GF(2^8).
//   xtime(a) = (a << 1) ^ (0x1b if a & 0x80 else 0)
// -----------------------------------------------------------------------
#define XTIME(src, dst, tmp) \
    VSLLB $1, src, dst; \     /* dst = src << 1 */
    VSRAB $7, src, tmp; \     /* tmp[i] = 0xff if src[i]>=0x80, else 0x00 */
    VANDB $0x1b, tmp, tmp; \  /* tmp[i] = 0x1b if src[i]>=0x80, else 0x00 */
    VXORV tmp, dst, dst       /* dst ^= tmp */

#define MIXCOLUMNS(src, out, r1, r2, r3, t1, t2, t3) \
    VSHUFB r1, src, src, t1; \      /* t1 = rot1(a) */
    VXORV  src, t1, t2; \           /* t2 = a ^ rot1(a) */
    VSHUFB r2, src, src, t3; \      /* t3 = rot2(a) */
    VXORV  t2, t3, out; \           /* out = a^rot1(a)^rot2(a) */
    VSHUFB r3, src, src, t3; \      /* t3 = rot3(a) */
    VXORV  out, t3, out; \          /* out = t = a0^a1^a2^a3（each byte） */
    XTIME(t2, t1, t3); \            /* t1 = xtime(a^rot1(a)) */
    VXORV  src, out, out; \         /* out = a ^ t */
    VXORV  out, t1, out             /* out ^= xtime(...) */

// func ctrBlocks1Asm(nr int, xk *[60]uint32, dst, src *[BlockSize]byte, ivlo, ivhi uint64)
TEXT ·ctrBlocks1Asm(SB), NOSPLIT, $16-48
    MOVV nr+0(FP),    R4
    MOVV xk+8(FP),    R5
    MOVV dst+16(FP),  R6
    MOVV src+24(FP),  R7
    MOVV ivlo+32(FP), R8
    MOVV ivhi+40(FP), R9

    // Construct a 128-bit counter block onto the stack, big endian [ivhi][ivlo]
    REVBV R9, R10          // ivhi little -> big
    REVBV R8, R11          // ivlo little -> big
    MOVV  R10, 0(R3)
    MOVV  R11, 8(R3)
    VMOVQ (R3), V0         // V0 = counter block

    MOVV $shiftRows(SB), R12
    VMOVQ (R12), V7
    MOVV $byteSwap32(SB), R12
    VMOVQ (R12), V2
    MOVV $rot1(SB), R12
    VMOVQ (R12), V10
    MOVV $rot2(SB), R12
    VMOVQ (R12), V11
    MOVV $rot3(SB), R12
    VMOVQ (R12), V12
    MOVV $sbox0_0(SB), R14

    // AddRoundKey
    VMOVQ (R5), V8
    ADDV  $16, R5
    VSHUFB V2, V8, V8, V8 // V8 = byte-swap each 32-bit word of round key
    VXORV  V8, V0, V0

    // middle round: nr-1
    MOVV R4, R12
    ADDV $-1, R12
Lenc_loop:
    SUBBYTES(V0, R14, V0)

    // ShiftRows
    VSHUFB V7, V0, V0, V0

    MIXCOLUMNS(V0, V1, V10, V11, V12, V3, V4, V5)

    // AddRoundKey
    VMOVQ (R5), V8
    ADDV  $16, R5
    VSHUFB V2, V8, V8, V8
    VXORV  V8, V1, V0

    ADDV $-1, R12
    BNE  R12, R0, Lenc_loop

    // last round: SubBytes + ShiftRows + AddRoundKey (no MixColumns)
    SUBBYTES(V0, R14, V0)
    VSHUFB V7, V0, V0, V0
    VMOVQ (R5), V8
    VSHUFB V2, V8, V8, V8
    VXORV V8, V0, V0

    VMOVQ (R7), V1
    VXORV V0, V1, V0
    VMOVQ V0, (R6)

    RET

// func ctrBlocks2Asm(nr int, xk *[60]uint32, dst, src *[2*BlockSize]byte, ivlo, ivhi uint64)
TEXT ·ctrBlocks2Asm(SB), NOSPLIT, $32-48
    MOVV nr+0(FP),    R4
    MOVV xk+8(FP),    R5
    MOVV dst+16(FP),  R6
    MOVV src+24(FP),  R7
    MOVV ivlo+32(FP), R8
    MOVV ivhi+40(FP), R9

    // Build counter 0 = (ivlo, ivhi)
    REVBV R9, R10
    REVBV R8, R11
    MOVV  R10, 0(R3)
    MOVV  R11, 8(R3)
    VMOVQ (R3), V0

    // Build counter 1 = (ivlo+1, ivhi + carry)
    MOVV  R8, R16
    ADDV  $1, R16, R17          // R17 = ivlo+1
    SGTU  R16, R17, R18         // R18 = 1 if overflow (unsigned wraparound), else 0
    ADDV  R9, R18, R19          // R19 = ivhi + carry
    REVBV R19, R10
    REVBV R17, R11
    MOVV  R10, 16(R3)
    MOVV  R11, 24(R3)
    VMOVQ 16(R3), V1

    // Load persistent constant tables (same as ctrBlocks1Asm)
    MOVV $shiftRows(SB), R12
    VMOVQ (R12), V7
    MOVV $byteSwap32(SB), R12
    VMOVQ (R12), V2
    MOVV $rot1(SB), R12
    VMOVQ (R12), V10
    MOVV $rot2(SB), R12
    VMOVQ (R12), V11
    MOVV $rot3(SB), R12
    VMOVQ (R12), V12
    MOVV $sbox0_0(SB), R14

    // Initial AddRoundKey (both blocks share the same first round key)
    VMOVQ (R5), V8
    ADDV  $16, R5
    VSHUFB V2, V8, V8, V8
    VXORV  V8, V0, V0
    VXORV  V8, V1, V1

    MOVV  R4, R12
    ADDV  $-1, R12          // R12 = nr-1 main rounds

Lenc2_loop:
    // Round key for this round (shared)
    VMOVQ (R5), V8
    ADDV  $16, R5
    VSHUFB V2, V8, V8, V8

    // Block 0
    SUBBYTES(V0, R14, V13)
    VSHUFB V7, V13, V13, V0
    MIXCOLUMNS(V0, V3, V10, V11, V12, V4, V5, V6)
    VXORV  V8, V3, V0

    // Block 1
    SUBBYTES(V1, R14, V13)
    VSHUFB V7, V13, V13, V1
    MIXCOLUMNS(V1, V3, V10, V11, V12, V4, V5, V6)
    VXORV  V8, V3, V1

    ADDV $-1, R12
    BNE  R12, R0, Lenc2_loop

    // Final round: SubBytes + ShiftRows + AddRoundKey (no MixColumns)
    VMOVQ (R5), V8
    VSHUFB V2, V8, V8, V8

    SUBBYTES(V0, R14, V13)
    VSHUFB V7, V13, V13, V0
    VXORV  V8, V0, V0

    SUBBYTES(V1, R14, V13)
    VSHUFB V7, V13, V13, V1
    VXORV  V8, V1, V1

    // XOR plaintext, store
    VMOVQ (R7),    V20
    VMOVQ 16(R7),  V21
    VXORV V0, V20, V0
    VXORV V1, V21, V1
    VMOVQ V0, (R6)
    VMOVQ V1, 16(R6)

    RET

// func ctrBlocks4Asm(nr int, xk *[60]uint32, dst, src *[4*BlockSize]byte, ivlo, ivhi uint64)
TEXT ·ctrBlocks4Asm(SB), NOSPLIT, $16-48
    MOVV nr+0(FP),    R4
    MOVV xk+8(FP),    R5
    MOVV dst+16(FP),  R6
    MOVV src+24(FP),  R7
    MOVV ivlo+32(FP), R8
    MOVV ivhi+40(FP), R9

    // Construct 4 consecutive counter blocks in sequence (big endian), ivlo+1 needs to process carry between blocks
    REVBV R9, R10
    REVBV R8, R11
    MOVV  R10, 0(R3)
    MOVV  R11, 8(R3)
    VMOVQ (R3), V0            // block 0

    ADDV  $1, R8, R16         // R16 = ivlo+1
    SGTU  R16, R8, R17        // R17 = 1 if not overflow (R16 > R8); if overflow R17=0
    XOR   $1, R17             // R17 = 1 overflow need carry
    ADDV  R17, R9, R18        // R18 = ivhi + carry
    REVBV R18, R10
    REVBV R16, R11
    MOVV  R10, 0(R3)
    MOVV  R11, 8(R3)
    VMOVQ (R3), V1            // block 1

    ADDV  $1, R16, R23
    SGTU  R23, R16, R17
    XOR   $1, R17
    ADDV  R17, R18, R24
    REVBV R24, R10
    REVBV R23, R11
    MOVV  R10, 0(R3)
    MOVV  R11, 8(R3)
    VMOVQ (R3), V3            // block 2

    ADDV  $1, R23, R25
    SGTU  R25, R23, R17
    XOR   $1, R17
    ADDV  R17, R24, R26
    REVBV R26, R10
    REVBV R25, R11
    MOVV  R10, 0(R3)
    MOVV  R11, 8(R3)
    VMOVQ (R3), V4            // block 3

    MOVV $shiftRows(SB), R12
    VMOVQ (R12), V7
    MOVV $byteSwap32(SB), R12
    VMOVQ (R12), V2
    MOVV $rot1(SB), R12
    VMOVQ (R12), V10
    MOVV $rot2(SB), R12
    VMOVQ (R12), V11
    MOVV $rot3(SB), R12
    VMOVQ (R12), V12
    MOVV $sbox0_0(SB), R14

    // Initial AddRoundKey (4 blocks share the same round key)
    VMOVQ (R5), V8
    VSHUFB V2, V8, V8, V8
    VXORV  V8, V0, V0
    VXORV  V8, V1, V1
    VXORV  V8, V3, V3
    VXORV  V8, V4, V4
    ADDV  $16, R5

    MOVV  R4, R15
    ADDV  $-1, R15

Lenc4_loop:
    SUBBYTES(V0, R14, V13)
    VSHUFB V7, V13, V13, V0
    MIXCOLUMNS(V0, V13, V10, V11, V12, V5, V6, V9)

    SUBBYTES(V1, R14, V14)
    VSHUFB V7, V14, V14, V1
    MIXCOLUMNS(V1, V14, V10, V11, V12, V5, V6, V9)

    SUBBYTES(V3, R14, V15)
    VSHUFB V7, V15, V15, V3
    MIXCOLUMNS(V3, V15, V10, V11, V12, V5, V6, V9)

    SUBBYTES(V4, R14, V16)
    VSHUFB V7, V16, V16, V4
    MIXCOLUMNS(V4, V16, V10, V11, V12, V5, V6, V9)

    VMOVQ (R5), V8
    VSHUFB V2, V8, V8, V8
    VXORV  V8, V13, V0
    VXORV  V8, V14, V1
    VXORV  V8, V15, V3
    VXORV  V8, V16, V4
    ADDV  $16, R5

    ADDV  $-1, R15
    BNE   R15, R0, Lenc4_loop

    // last round: SubBytes + ShiftRows + AddRoundKey (no MixColumns)
    VMOVQ (R5), V8
    VSHUFB V2, V8, V8, V8

    SUBBYTES(V0, R14, V13)
    VSHUFB V7, V13, V13, V0
    VXORV  V8, V0, V0

    SUBBYTES(V1, R14, V13)
    VSHUFB V7, V13, V13, V1
    VXORV  V8, V1, V1

    SUBBYTES(V3, R14, V13)
    VSHUFB V7, V13, V13, V3
    VXORV  V8, V3, V3

    SUBBYTES(V4, R14, V13)
    VSHUFB V7, V13, V13, V4
    VXORV  V8, V4, V4

    VMOVQ (R7),   V14
    VMOVQ 16(R7), V15
    VMOVQ 32(R7), V16
    VMOVQ 48(R7), V17
    VXORV V0, V14, V14
    VXORV V1, V15, V15
    VXORV V3, V16, V16
    VXORV V4, V17, V17
    VMOVQ V14, (R6)
    VMOVQ V15, 16(R6)
    VMOVQ V16, 32(R6)
    VMOVQ V17, 48(R6)

    RET

// func ctrBlocks8Asm(nr int, xk *[60]uint32, dst, src *[8*BlockSize]byte, ivlo, ivhi uint64)
TEXT ·ctrBlocks8Asm(SB), NOSPLIT, $32-48
    MOVV nr+0(FP),    R4
    MOVV xk+8(FP),    R5
    MOVV dst+16(FP),  R6
    MOVV src+24(FP),  R7
    MOVV ivlo+32(FP), R8
    MOVV ivhi+40(FP), R9

    REVBV R9, R10
    REVBV R8, R11
    MOVV  R10, 0(R3)
    MOVV  R11, 8(R3)
    VMOVQ (R3), V0

    ADDV  $1, R8, R16
    SGTU  R8, R16, R17
    ADDV  R17, R9, R17
    MOVV  R16, R8
    MOVV  R17, R9
    REVBV R9, R10
    REVBV R8, R11
    MOVV  R10, 16(R3)
    MOVV  R11, 24(R3)
    VMOVQ 16(R3), V1

    ADDV  $1, R8, R16
    SGTU  R8, R16, R17
    ADDV  R17, R9, R17
    MOVV  R16, R8
    MOVV  R17, R9
    REVBV R9, R10
    REVBV R8, R11
    MOVV  R10, 0(R3)
    MOVV  R11, 8(R3)
    VMOVQ (R3), V3

    ADDV  $1, R8, R16
    SGTU  R8, R16, R17
    ADDV  R17, R9, R17
    MOVV  R16, R8
    MOVV  R17, R9
    REVBV R9, R10
    REVBV R8, R11
    MOVV  R10, 16(R3)
    MOVV  R11, 24(R3)
    VMOVQ 16(R3), V4

    ADDV  $1, R8, R16
    SGTU  R8, R16, R17
    ADDV  R17, R9, R17
    MOVV  R16, R8
    MOVV  R17, R9
    REVBV R9, R10
    REVBV R8, R11
    MOVV  R10, 0(R3)
    MOVV  R11, 8(R3)
    VMOVQ (R3), V5

    ADDV  $1, R8, R16
    SGTU  R8, R16, R17
    ADDV  R17, R9, R17
    MOVV  R16, R8
    MOVV  R17, R9
    REVBV R9, R10
    REVBV R8, R11
    MOVV  R10, 16(R3)
    MOVV  R11, 24(R3)
    VMOVQ 16(R3), V6

    ADDV  $1, R8, R16
    SGTU  R8, R16, R17
    ADDV  R17, R9, R17
    MOVV  R16, R8
    MOVV  R17, R9
    REVBV R9, R10
    REVBV R8, R11
    MOVV  R10, 0(R3)
    MOVV  R11, 8(R3)
    VMOVQ (R3), V8

    ADDV  $1, R8, R16
    SGTU  R8, R16, R17
    ADDV  R17, R9, R17
    MOVV  R16, R8
    MOVV  R17, R9
    REVBV R9, R10
    REVBV R8, R11
    MOVV  R10, 16(R3)
    MOVV  R11, 24(R3)
    VMOVQ 16(R3), V9

    MOVV $shiftRows(SB), R12
    VMOVQ (R12), V7
    MOVV $byteSwap32(SB), R12
    VMOVQ (R12), V2
    MOVV $rot1(SB), R12
    VMOVQ (R12), V10
    MOVV $rot2(SB), R12
    VMOVQ (R12), V11
    MOVV $rot3(SB), R12
    VMOVQ (R12), V12
    MOVV $sbox0_0(SB), R14

    VMOVQ (R5), V15
    ADDV  $16, R5
    VSHUFB V2, V15, V15, V15
    VXORV  V15, V0, V0
    VXORV  V15, V1, V1
    VXORV  V15, V3, V3
    VXORV  V15, V4, V4
    VXORV  V15, V5, V5
    VXORV  V15, V6, V6
    VXORV  V15, V8, V8
    VXORV  V15, V9, V9

    MOVV R4, R13
    ADDV $-1, R13

Lenc8_loop:
    // SubBytes + ShiftRows + MixColumns + AddRoundKey, Process 8 state blocks in sequence 
    SUBBYTES(V0, R14, V0)
    SUBBYTES(V1, R14, V1)
    SUBBYTES(V3, R14, V3)
    SUBBYTES(V4, R14, V4)
    SUBBYTES(V5, R14, V5)
    SUBBYTES(V6, R14, V6)
    SUBBYTES(V8, R14, V8)
    SUBBYTES(V9, R14, V9)

    VSHUFB V7, V0, V0, V0
    VSHUFB V7, V1, V1, V1
    VSHUFB V7, V3, V3, V3
    VSHUFB V7, V4, V4, V4
    VSHUFB V7, V5, V5, V5
    VSHUFB V7, V6, V6, V6
    VSHUFB V7, V8, V8, V8
    VSHUFB V7, V9, V9, V9

    MIXCOLUMNS(V0, V23, V10, V11, V12, V20, V21, V22)
    MIXCOLUMNS(V1, V24, V10, V11, V12, V20, V21, V22)
    MIXCOLUMNS(V3, V25, V10, V11, V12, V20, V21, V22)
    MIXCOLUMNS(V4, V26, V10, V11, V12, V20, V21, V22)
    MIXCOLUMNS(V5, V27, V10, V11, V12, V20, V21, V22)
    MIXCOLUMNS(V6, V28, V10, V11, V12, V20, V21, V22)
    MIXCOLUMNS(V8, V29, V10, V11, V12, V20, V21, V22)
    MIXCOLUMNS(V9, V30, V10, V11, V12, V20, V21, V22)

    VMOVQ (R5), V15
    ADDV  $16, R5
    VSHUFB V2, V15, V15, V15
    VXORV  V15, V23, V0
    VXORV  V15, V24, V1
    VXORV  V15, V25, V3
    VXORV  V15, V26, V4
    VXORV  V15, V27, V5
    VXORV  V15, V28, V6
    VXORV  V15, V29, V8
    VXORV  V15, V30, V9

    ADDV $-1, R13
    BNE  R13, R0, Lenc8_loop

    // last round: SubBytes + ShiftRows + AddRoundKey (no MixColumns)
    SUBBYTES(V0, R14, V0)
    SUBBYTES(V1, R14, V1)
    SUBBYTES(V3, R14, V3)
    SUBBYTES(V4, R14, V4)
    SUBBYTES(V5, R14, V5)
    SUBBYTES(V6, R14, V6)
    SUBBYTES(V8, R14, V8)
    SUBBYTES(V9, R14, V9)

    VSHUFB V7, V0, V0, V0
    VSHUFB V7, V1, V1, V1
    VSHUFB V7, V3, V3, V3
    VSHUFB V7, V4, V4, V4
    VSHUFB V7, V5, V5, V5
    VSHUFB V7, V6, V6, V6
    VSHUFB V7, V8, V8, V8
    VSHUFB V7, V9, V9, V9

    VMOVQ (R5), V15
    VSHUFB V2, V15, V15, V15
    VXORV  V15, V0, V0
    VXORV  V15, V1, V1
    VXORV  V15, V3, V3
    VXORV  V15, V4, V4
    VXORV  V15, V5, V5
    VXORV  V15, V6, V6
    VXORV  V15, V8, V8
    VXORV  V15, V9, V9

    VMOVQ (0*16)(R7), V16;  VXORV V0, V16, V16;  VMOVQ V16, (0*16)(R6)
    VMOVQ (1*16)(R7), V17;  VXORV V1, V17, V17;  VMOVQ V17, (1*16)(R6)
    VMOVQ (2*16)(R7), V18;  VXORV V3, V18, V18;  VMOVQ V18, (2*16)(R6)
    VMOVQ (3*16)(R7), V19;  VXORV V4, V19, V19;  VMOVQ V19, (3*16)(R6)
    VMOVQ (4*16)(R7), V20;  VXORV V5, V20, V20;  VMOVQ V20, (4*16)(R6)
    VMOVQ (5*16)(R7), V21;  VXORV V6, V21, V21;  VMOVQ V21, (5*16)(R6)
    VMOVQ (6*16)(R7), V22;  VXORV V8, V22, V22;  VMOVQ V22, (6*16)(R6)
    VMOVQ (7*16)(R7), V23;  VXORV V9, V23, V23;  VMOVQ V23, (7*16)(R6)

    RET
