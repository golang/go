// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// AES for loong64 using LSX VSHUFB-based software S-box lookup.
// No hardware AES instructions exist on LoongArch (as of LA464/LA664).
//
// SubBytes: 256-byte S-box split into 8×32-byte chunks; 8 VSHUFB ops
//           select the right byte per input nibble group.
// ShiftRows: implemented via a VSHUFB with a fixed permutation table.
// MixColumns: GF(2^8) multiply-by-2 via shift+conditional-XOR, then
//             combine columns with VXORV.
// InvMixColumns: implemented via GFMULCONST, a nibble-split VSHUFB table
//   lookup (separate lo/hi 16-entry tables per GF(2^8) constant). The
//   9/11/13/14 constant multiples are computed sequentially — the four
//   GFMULCONST calls share the same midx1/midx2/tlo scratch registers,
//   so they are NOT computed in parallel — then combined with
//   rot1/rot2/rot3 (VSHUFB) and VXORV. This replaces an earlier
//   XTIME-chain implementation (three chained GF(2^8) doublings), which
//   was measured to be slower on AESCBCDecrypt1K/Decrypt benchmarks on
//   this target despite a theoretically shorter critical path; see the
//   GFMULCONST-table generation note below for how the lo/hi constant
//   tables were produced and verified.
//
// expandKeyAsm: Phase 1/2 (deriving enc[] from the input key) is a
//               scalar port of expandKeyGeneric — the key schedule has
//               a true serial dependency chain and operates on 32-bit
//               words smaller than the vector width, so it is not
//               vectorized. Phase 3 (deriving dec[] from enc[]) has
//               independent iterations over 16-byte word groups and is
//               vectorized using INVMIXCOLUMNS directly (no SubBytes —
//               the scalar td0/td1/td2/td3 tables already fold in and
//               cancel out sbox0/sbox1, see aes_generic.go), with a
//               byte-order swap (VSHUFB with byteSwap32) before/after
//               to convert between word-storage order and the state
//               byte order that INVMIXCOLUMNS's rot1/rot2/rot3 tables
//               assume.

//go:build !purego

#include "textflag.h"

// AES S-box tables for loong64 scalar implementation.
// Stored as raw byte arrays; each DATA line encodes 8 consecutive
// sbox bytes as a little-endian uint64 (byte[0] in the lowest address).
//
// Formula: value = b7<<56 | b6<<48 | b5<<40 | b4<<32
//                | b3<<24 | b2<<16 | b1<<8  | b0

// -----------------------------------------------------------------------
// sbox0 — AES encryption S-box (256 bytes)
// Source: FIPS-197 Figure 7 / const.go var sbox0
// -----------------------------------------------------------------------
DATA	sbox0_0+0x00(SB)/8, $0xc56f6bf27b777c63		// sbox0[  0.. 7]
DATA	sbox0_0+0x08(SB)/8, $0x76abd7fe2b670130		// sbox0[  8..15]
DATA	sbox0_0+0x10(SB)/8, $0xf04759fa7dc982ca		// sbox0[ 16..23]
DATA	sbox0_0+0x18(SB)/8, $0xc072a49cafa2d4ad		// sbox0[ 24..31]
DATA	sbox0_0+0x20(SB)/8, $0xccf73f362693fdb7		// sbox0[ 32..39]
DATA	sbox0_0+0x28(SB)/8, $0x1531d871f1e5a534		// sbox0[ 40..47]
DATA	sbox0_0+0x30(SB)/8, $0x9a059618c323c704		// sbox0[ 48..55]
DATA	sbox0_0+0x38(SB)/8, $0x75b227ebe2801207		// sbox0[ 56..63]
DATA	sbox0_0+0x40(SB)/8, $0xa05a6e1b1a2c8309		// sbox0[ 64..71]
DATA	sbox0_0+0x48(SB)/8, $0x842fe329b3d63b52		// sbox0[ 72..79]
DATA	sbox0_0+0x50(SB)/8, $0x5bb1fc20ed00d153		// sbox0[ 80..87]
DATA	sbox0_0+0x58(SB)/8, $0xcf584c4a39becb6a		// sbox0[ 88..95]
DATA	sbox0_0+0x60(SB)/8, $0x85334d43fbaaefd0		// sbox0[ 96..103]
DATA	sbox0_0+0x68(SB)/8, $0xa89f3c507f02f945		// sbox0[104..111]
DATA	sbox0_0+0x70(SB)/8, $0xf5389d928f40a351		// sbox0[112..119]
DATA	sbox0_0+0x78(SB)/8, $0xd2f3ff1021dab6bc		// sbox0[120..127]
DATA	sbox0_0+0x80(SB)/8, $0x1744975fec130ccd		// sbox0[128..135]
DATA	sbox0_0+0x88(SB)/8, $0x73195d643d7ea7c4		// sbox0[136..143]
DATA	sbox0_0+0x90(SB)/8, $0x88902a22dc4f8160		// sbox0[144..151]
DATA	sbox0_0+0x98(SB)/8, $0xdb0b5ede14b8ee46		// sbox0[152..159]
DATA	sbox0_0+0xa0(SB)/8, $0x5c2406490a3a32e0		// sbox0[160..167]
DATA	sbox0_0+0xa8(SB)/8, $0x79e4959162acd3c2		// sbox0[168..175]
DATA	sbox0_0+0xb0(SB)/8, $0xa94ed58d6d37c8e7		// sbox0[176..183]
DATA	sbox0_0+0xb8(SB)/8, $0x08ae7a65eaf4566c		// sbox0[184..191]
DATA	sbox0_0+0xc0(SB)/8, $0xc6b4a61c2e2578ba		// sbox0[192..199]
DATA	sbox0_0+0xc8(SB)/8, $0x8a8bbd4b1f74dde8		// sbox0[200..207]
DATA	sbox0_0+0xd0(SB)/8, $0x0ef6034866b53e70		// sbox0[208..215]
DATA	sbox0_0+0xd8(SB)/8, $0x9e1dc186b9573561		// sbox0[216..223]
DATA	sbox0_0+0xe0(SB)/8, $0x948ed9691198f8e1		// sbox0[224..231]
DATA	sbox0_0+0xe8(SB)/8, $0xdf2855cee9871e9b		// sbox0[232..239]
DATA	sbox0_0+0xf0(SB)/8, $0x6842e6bf0d89a18c		// sbox0[240..247]
DATA	sbox0_0+0xf8(SB)/8, $0x16bb54b00f2d9941		// sbox0[248..255]
GLOBL	sbox0_0(SB), (NOPTR+RODATA), $256

// -----------------------------------------------------------------------
// sbox1 — AES decryption (inverse) S-box (256 bytes)
// Source: FIPS-197 Figure 14 / const.go var sbox1
// -----------------------------------------------------------------------
DATA	sbox1_0<>+0x00(SB)/8, $0x38a53630d56a0952	// sbox1[  0.. 7]
DATA	sbox1_0<>+0x08(SB)/8, $0xfbd7f3819ea340bf	// sbox1[  8..15]
DATA	sbox1_0<>+0x10(SB)/8, $0x87ff2f9b8239e37c	// sbox1[ 16..23]
DATA	sbox1_0<>+0x18(SB)/8, $0xcbe9dec444438e34	// sbox1[ 24..31]
DATA	sbox1_0<>+0x20(SB)/8, $0x3d23c2a632947b54	// sbox1[ 32..39]
DATA	sbox1_0<>+0x28(SB)/8, $0x4ec3fa420b954cee	// sbox1[ 40..47]
DATA	sbox1_0<>+0x30(SB)/8, $0xb224d92866a12e08	// sbox1[ 48..55]
DATA	sbox1_0<>+0x38(SB)/8, $0x25d18b6d49a25b76	// sbox1[ 56..63]
DATA	sbox1_0<>+0x40(SB)/8, $0x1698688664f6f872	// sbox1[ 64..71]
DATA	sbox1_0<>+0x48(SB)/8, $0x92b6655dcc5ca4d4	// sbox1[ 72..79]
DATA	sbox1_0<>+0x50(SB)/8, $0xdab9edfd5048706c	// sbox1[ 80..87]
DATA	sbox1_0<>+0x58(SB)/8, $0x849d8da75746155e	// sbox1[ 88..95]
DATA	sbox1_0<>+0x60(SB)/8, $0x0ad3bc8c00abd890	// sbox1[ 96..103]
DATA	sbox1_0<>+0x68(SB)/8, $0x0645b3b80558e4f7	// sbox1[104..111]
DATA	sbox1_0<>+0x70(SB)/8, $0x020f3fca8f1e2cd0	// sbox1[112..119]
DATA	sbox1_0<>+0x78(SB)/8, $0x6b8a130103bdafc1	// sbox1[120..127]
DATA	sbox1_0<>+0x80(SB)/8, $0xeadc674f4111913a	// sbox1[128..135]
DATA	sbox1_0<>+0x88(SB)/8, $0x73e6b4f0cecff297	// sbox1[136..143]
DATA	sbox1_0<>+0x90(SB)/8, $0x8535ade72274ac96	// sbox1[144..151]
DATA	sbox1_0<>+0x98(SB)/8, $0x6edf751ce837f9e2	// sbox1[152..159]
DATA	sbox1_0<>+0xa0(SB)/8, $0x89c5291d711af147	// sbox1[160..167]
DATA	sbox1_0<>+0xa8(SB)/8, $0x1bbe18aa0e62b76f	// sbox1[168..175]
DATA	sbox1_0<>+0xb0(SB)/8, $0x2079d2c64b3e56fc	// sbox1[176..183]
DATA	sbox1_0<>+0xb8(SB)/8, $0xf45acd78fec0db9a	// sbox1[184..191]
DATA	sbox1_0<>+0xc0(SB)/8, $0x31c7078833a8dd1f	// sbox1[192..199]
DATA	sbox1_0<>+0xc8(SB)/8, $0x5fec8027591012b1	// sbox1[200..207]
DATA	sbox1_0<>+0xd0(SB)/8, $0x0d4ab519a97f5160	// sbox1[208..215]
DATA	sbox1_0<>+0xd8(SB)/8, $0xef9cc9939f7ae52d	// sbox1[216..223]
DATA	sbox1_0<>+0xe0(SB)/8, $0xb0f52aae4d3be0a0	// sbox1[224..231]
DATA	sbox1_0<>+0xe8(SB)/8, $0x619953833cbbebc8	// sbox1[232..239]
DATA	sbox1_0<>+0xf0(SB)/8, $0x26d677ba7e042b17	// sbox1[240..247]
DATA	sbox1_0<>+0xf8(SB)/8, $0x7d0c2155631469e1	// sbox1[248..255]
GLOBL	sbox1_0<>(SB), (NOPTR+RODATA), $256

// ShiftRows permutation for encryption:
//   state byte positions after ShiftRows (row i rotated left by i):
//   row0: 0,1,2,3  row1: 5,6,7,4  row2: 10,11,8,9  row3: 15,12,13,14
DATA	shiftRows+0x00(SB)/8, $0x030e09040f0a0500
DATA	shiftRows+0x08(SB)/8, $0x0b06010c07020d08
GLOBL	shiftRows(SB), (NOPTR+RODATA), $16

// rot1: {1,2,3,0, 5,6,7,4, 9,10,11,8, 13,14,15,12}
DATA	rot1+0x00(SB)/8, $0x0407060500030201
DATA	rot1+0x08(SB)/8, $0x0c0f0e0d080b0a09
GLOBL	rot1(SB), (NOPTR+RODATA), $16

// rot2: {2,3,0,1, 6,7,4,5, 10,11,8,9, 14,15,12,13}
DATA	rot2+0x00(SB)/8, $0x0504070601000302
DATA	rot2+0x08(SB)/8, $0x0d0c0f0e09080b0a
GLOBL	rot2(SB), (NOPTR+RODATA), $16

// rot3: {3,0,1,2, 7,4,5,6, 11,8,9,10, 15,12,13,14}
DATA	rot3+0x00(SB)/8, $0x0605040702010003
DATA	rot3+0x08(SB)/8, $0x0e0d0c0f0a09080b
GLOBL	rot3(SB), (NOPTR+RODATA), $16

// InvShiftRows permutation for decryption.
DATA	invShiftRows<>+0x00(SB)/8, $0x0b0e0104070a0d00
DATA	invShiftRows<>+0x08(SB)/8, $0x0306090c0f020508
GLOBL	invShiftRows<>(SB), (NOPTR+RODATA), $16

DATA	byteSwap32+0x00(SB)/8, $0x0405060700010203	// index 0-7:  {3,2,1,0,7,6,5,4}
DATA	byteSwap32+0x08(SB)/8, $0x0c0d0e0f08090a0b	// index 8-15: {11,10,9,8,15,14,13,12}
GLOBL	byteSwap32(SB), (NOPTR+RODATA), $16

// mul9lo[v] = mul(9, v), v = 0..15
DATA	mul9lo<>+0x00(SB)/8, $0x3f362d241b120900
DATA	mul9lo<>+0x08(SB)/8, $0x777e656c535a4148
GLOBL	mul9lo<>(SB), (NOPTR+RODATA), $16

// mul9hi[v] = mul(9, v<<4), v = 0..15
DATA	mul9hi<>+0x00(SB)/8, $0xdd4de676ab3b9000
DATA	mul9hi<>+0x08(SB)/8, $0x31a10a9a47d77cec
GLOBL	mul9hi<>(SB), (NOPTR+RODATA), $16

// mul11lo[v] = mul(11, v), v = 0..15
DATA	mul11lo<>+0x00(SB)/8, $0x313a272c1d160b00
DATA	mul11lo<>+0x08(SB)/8, $0x69627f74454e5358
GLOBL	mul11lo<>(SB), (NOPTR+RODATA), $16

// mul11hi[v] = mul(11, v<<4), v = 0..15
DATA	mul11hi<>+0x00(SB)/8, $0x3d8d46f6cb7bb000
DATA	mul11hi<>+0x08(SB)/8, $0xca7ab1013c8c47f7
GLOBL	mul11hi<>(SB), (NOPTR+RODATA), $16

// mul13lo[v] = mul(13, v), v = 0..15
DATA	mul13lo<>+0x00(SB)/8, $0x232e3934171a0d00
DATA	mul13lo<>+0x08(SB)/8, $0x4b46515c7f726568
GLOBL	mul13lo<>(SB), (NOPTR+RODATA), $16

// mul13hi[v] = mul(13, v<<4), v = 0..15
DATA	mul13hi<>+0x00(SB)/8, $0x06d6bd6d6bbbd000
DATA	mul13hi<>+0x08(SB)/8, $0xdc0c67b7b1610ada
GLOBL	mul13hi<>(SB), (NOPTR+RODATA), $16

// mul14lo[v] = mul(14, v), v = 0..15
DATA	mul14lo<>+0x00(SB)/8, $0x2a243638121c0e00
DATA	mul14lo<>+0x08(SB)/8, $0x5a544648626c7e70
GLOBL	mul14lo<>(SB), (NOPTR+RODATA), $16

// mul14hi[v] = mul(14, v<<4), v = 0..15
DATA	mul14hi<>+0x00(SB)/8, $0x96764dad3bdbe000
DATA	mul14hi<>+0x08(SB)/8, $0xd7370cec7a9aa141
GLOBL	mul14hi<>(SB), (NOPTR+RODATA), $16

#define SUBBYTES(src, sbox_ptr, out) \
	VANDB	$0xe0, src, V25;	\  // V25 = src & 0xe0, keep only bit7/bit6/bit5
	VSRAB	$7, V25, V31;		\  // V31 = byte-wise broadcast of bit7 (0xFF/0x00)
	VSLLB	$1, V25, V24;		\  // shift bit6 into the sign position
	VSRAB	$7, V24, V30;		\  // V30 = byte-wise broadcast of bit6
	VSLLB	$2, V25, V24;		\  // shift bit5 into the sign position
	VSRAB	$7, V24, V29;		\  // V29 = byte-wise broadcast of bit5
	VANDB	$0x1f, src, V26;	\  // V26 = src & 0x1f, index for VSHUFB 32-way lookup
	\
	/* ---- 2. Chunk0/1 -> r01 (V16..V19,V28,V27 freed after use) ---- */ \
	VMOVQ	(sbox_ptr), V16;   \
	VMOVQ	16(sbox_ptr), V17;   \
	VMOVQ	32(sbox_ptr), V18;   \
	VMOVQ	48(sbox_ptr), V19;   \
	VSHUFB	V26, V16, V17, V28;  \		// V28 = chunk0_raw
	VSHUFB	V26, V18, V19, V27;  \		// V27 = chunk1_raw
	VBITSELV	V29, V27, V28, V20; \	// V20 = r01 = M5 ? chunk1 : chunk0
	\
	/* ---- 3. Chunk2/3 -> r23 ---- */ \
	VMOVQ	64(sbox_ptr), V16;  \
	VMOVQ	80(sbox_ptr), V17;  \
	VMOVQ	96(sbox_ptr), V18;  \
	VMOVQ	112(sbox_ptr), V19;  \
	VSHUFB	V26, V16, V17, V28;  \		// V28 = chunk2_raw
	VSHUFB	V26, V18, V19, V27;  \		// V27 = chunk3_raw
	VBITSELV	V29, V27, V28, V21; \	// V21 = r23
	\
	/* ---- 4. Merge first group (chunk0..3) -> r0123 ---- */ \
	VBITSELV	V30, V21, V20, V22; \	// V22 = r0123 = M6 ? r23 : r01
	\
	/* ---- 5. Chunk4/5 -> r45 ---- */ \
	VMOVQ	128(sbox_ptr), V16;  \
	VMOVQ	144(sbox_ptr), V17;  \
	VMOVQ	160(sbox_ptr), V18;  \
	VMOVQ	176(sbox_ptr), V19;  \
	VSHUFB	V26, V16, V17, V28;  \		// V28 = chunk4_raw
	VSHUFB	V26, V18, V19, V27;  \		// V27 = chunk5_raw
	VBITSELV	V29, V27, V28, V20; \	// V20 reused = r45
	\
	/* ---- 6. Chunk6/7 -> r67 ---- */ \
	VMOVQ	192(sbox_ptr), V16;  \
	VMOVQ	208(sbox_ptr), V17;  \
	VMOVQ	224(sbox_ptr), V18;  \
	VMOVQ	240(sbox_ptr), V19;  \
	VSHUFB	V26, V16, V17, V28;  \		// V28 = chunk6_raw
	VSHUFB	V26, V18, V19, V27;  \		// V27 = chunk7_raw
	VBITSELV	V29, V27, V28, V21; \	// V21 reused = r67
	\
	/* ---- 7. Merge second group (chunk4..7) -> r4567 ---- */ \
	VBITSELV	V30, V21, V20, V23; \	// V23 = r4567 = M6 ? r67 : r45
	\
	/* ---- 8. Final top-level merge ---- */ \
	VBITSELV	V31, V23, V22, out	// out = M7 ? r4567 : r0123

// -----------------------------------------------------------------------
// MixColumns helper: multiply each byte of Vn by 2 in GF(2^8).
//   xtime(a) = (a << 1) ^ (0x1b if a & 0x80 else 0)
// -----------------------------------------------------------------------
#define XTIME(src, dst, tmp) \
	VSLLB	$1, src, dst; \		/* dst = src << 1 */
	VSRAB	$7, src, tmp; \		/* tmp[i] = 0xff if src[i]>=0x80, else 0x00 */
	VANDB	$0x1b, tmp, tmp; \	/* tmp[i] = 0x1b if src[i]>=0x80, else 0x00 */
	VXORV	tmp, dst, dst		/* dst ^= tmp */

#define MIXCOLUMNS(src, out, r1, r2, r3, t1, t2, t3) \
	VSHUFB	r1, src, src, t1; \	/* t1 = rot1(a) */
	VXORV	src, t1, t2; \		/* t2 = a ^ rot1(a) */
	VSHUFB	r2, src, src, t3; \	/* t3 = rot2(a) */
	VXORV	t2, t3, out; \		/* out = a^rot1(a)^rot2(a) */
	VSHUFB	r3, src, src, t3; \	/* t3 = rot3(a) */
	VXORV	out, t3, out; \		/* out = t = a0^a1^a2^a3（each byte） */
	XTIME(t2, t1, t3); \		/* t1 = xtime(a^rot1(a)) */
	VXORV	src, out, out; \	/* out = a ^ t */
	VXORV	out, t1, out		/* out ^= xtime(...) */

// t = subw(t): S-box substitution for each byte of the 32-bit word t, results written back to t
#define SUBW_INLINE(t, sbox, tmp1, tmp2) \
	SRLV	$24, t, tmp1; \
	AND	$0xff, tmp1; \
	ADDV	sbox, tmp1, tmp1; \
	MOVBU	(tmp1), tmp1; \
	SLLV	$24, tmp1, tmp2; \
	SRLV	$16, t, tmp1; \
	AND	$0xff, tmp1; \
	ADDV	sbox, tmp1, tmp1; \
	MOVBU	(tmp1), tmp1; \
	SLLV	$16, tmp1, tmp1; \
	OR	tmp1, tmp2; \
	SRLV	$8, t, tmp1; \
	AND	$0xff, tmp1; \
	ADDV	sbox, tmp1, tmp1; \
	MOVBU	(tmp1), tmp1; \
	SLLV	$8, tmp1, tmp1; \
	OR	tmp1, tmp2; \
	AND	$0xff, t, tmp1; \
	ADDV	sbox, tmp1, tmp1; \
	MOVBU	(tmp1), tmp1; \
	OR	tmp1, tmp2; \
	MOVV	tmp2, t

TEXT ·encryptBlockAsm(SB), NOSPLIT, $0-32
	MOVV	nr+0(FP), R4
	MOVV	xk+8(FP), R5
	MOVV	dst+16(FP), R6
	MOVV	src+24(FP), R7

	// Load 16-byte state
	VMOVQ	(R7), V0

	// Load constant table addresses
	MOVV	$sbox0_0(SB), R9
	MOVV	$shiftRows(SB), R10
	MOVV	$rot1(SB), R11
	MOVV	$rot2(SB), R12
	MOVV	$rot3(SB), R13
	MOVV	$byteSwap32(SB), R14

	// Load vector constants
	VMOVQ	(R10), V7	// shiftRows index
	VMOVQ	(R11), V4	// rot1 index
	VMOVQ	(R12), V5	// rot2 index
	VMOVQ	(R13), V6	// rot3 index
	VMOVQ	(R14), V9	// byteSwap32 index

	// AddRoundKey (round 0)
	VMOVQ	(R5), V8
	VSHUFB	V9, V8, V8, V8
	VXORV	V8, V0, V0
	ADDV	$16, R5

	// Middle rounds: nr - 1 iterations
	ADDV	$-1, R4, R8
Lenc_loop:
	SUBBYTES(V0, R9, V0)

	// ShiftRows
	VSHUFB	V7, V0, V0, V0

	// MixColumns
	MIXCOLUMNS(V0, V15, V4, V5, V6, V10, V11, V12)

	// AddRoundKey
	VMOVQ	(R5), V8
	VSHUFB	V9, V8, V8, V8
	VXORV	V8, V15, V0
	ADDV	$16, R5

	SUBV	$1, R8
	BNE	R8, R0, Lenc_loop

	// Final round: SubBytes + ShiftRows + AddRoundKey (no MixColumns)
	SUBBYTES(V0, R9, V0)
	VSHUFB	V7, V0, V0, V0
	VMOVQ	(R5), V8
	VSHUFB	V9, V8, V8, V8
	VXORV	V8, V0, V0

	// Store ciphertext
	VMOVQ	V0, (R6)
	RET

// GFMULCONST(src, lo_tbl_ptr, hi_tbl_ptr, out, lo_idx, hi_idx, tlo)
// out = const * src  (GF(2^8)), via nibble-split lookup.
// lo_tbl_ptr/hi_tbl_ptr: base address registers for this constant's 16B tables.
// lo_idx/hi_idx/tlo: scratch, freed immediately after the macro.
#define GFMULCONST(src, lo_tbl, hi_tbl, out, lo_idx, hi_idx, tlo) \
	VANDB	$0x0f, src, lo_idx;      \	// lo_idx = src & 0x0f
	VSRLB	$4,   src, hi_idx;       \	// hi_idx = src >> 4 (unsigned, top nibble)
	VMOVQ	(lo_tbl), tlo;           \	// load this constant's lo table (16B)
	VSHUFB	lo_idx, tlo, tlo, lo_idx;\	// lo_idx = lo_table[src&0xf]  (single-source 16-way lookup)
	VMOVQ	(hi_tbl), tlo;           \	// load this constant's hi table (16B), tlo reused
	VSHUFB	hi_idx, tlo, tlo, hi_idx;\	// hi_idx = hi_table[src>>4]
	VXORV	lo_idx, hi_idx, out		// out = lo_part ^ hi_part

// INVMIXCOLUMNS(src, out, r1, r2, r3, t9, t11, t13, t14, midx1, midx2, tlo)
// r1/r2/r3: rotation index tables (already loaded, e.g. V4/V5/V6)
// t9/t11/t13/t14: four accumulator registers, each holds one const*src result
// midx1/midx2/tlo: shared scratch reused sequentially across the 4 GFMULCONST calls
#define INVMIXCOLUMNS(src, out, r1, r2, r3, t9, t11, t13, t14, midx1, midx2, tlo) \
	MOVV	$mul9lo<>(SB), R13; \
	MOVV	$mul9hi<>(SB), R14; \
	GFMULCONST(src, R13, R14, t9,  midx1, midx2, tlo); \	// t9  = 9*src
	\
	MOVV	$mul11lo<>(SB), R13; \
	MOVV	$mul11hi<>(SB), R14; \
	GFMULCONST(src, R13, R14, t11, midx1, midx2, tlo); \	// t11 = 11*src
	\
	MOVV	$mul13lo<>(SB), R13; \
	MOVV	$mul13hi<>(SB), R14; \
	GFMULCONST(src, R13, R14, t13, midx1, midx2, tlo); \	// t13 = 13*src
	\
	MOVV	$mul14lo<>(SB), R13; \
	MOVV	$mul14hi<>(SB), R14; \
	GFMULCONST(src, R13, R14, t14, midx1, midx2, tlo); \	// t14 = 14*src
	\
	/* out = 14*src ^ rot1(11*src) ^ rot2(13*src) ^ rot3(9*src) */ \
	VSHUFB	r1, t11, t11, t11; \	// t11 = rot1(11*src)
	VSHUFB	r2, t13, t13, t13; \	// t13 = rot2(13*src)
	VSHUFB	r3, t9,  t9,  t9;  \	// t9  = rot3(9*src)
	VXORV	t11, t14, out;     \	// out = 14*src ^ rot1(11*src)
	VXORV	t13, out, out;     \	// out ^= rot2(13*src)
	VXORV	t9,  out, out		// out ^= rot3(9*src)

TEXT ·decryptBlockAsm(SB), NOSPLIT, $0-32
	MOVV	nr+0(FP), R4
	MOVV	xk+8(FP), R5
	MOVV	dst+16(FP), R6
	MOVV	src+24(FP), R7

	// Load state
	VMOVQ	(R7), V0

	// Load invShiftRows index vector
	MOVV	$invShiftRows<>(SB), R9
	VMOVQ	(R9), V7

	// Load sbox1 base for InvSubBytes
	MOVV	$sbox1_0<>(SB), R10

	// Load byte-swap table for round-key endianness
	MOVV	$byteSwap32(SB), R11
	VMOVQ	(R11), V2

	// Load rotation indices for InvMixColumns
	MOVV	$rot1(SB), R12
	VMOVQ	(R12), V4
	MOVV	$rot2(SB), R12
	VMOVQ	(R12), V5
	MOVV	$rot3(SB), R12
	VMOVQ	(R12), V6

	// Initial AddRoundKey (last round key for decryption)
	VMOVQ	(R5), V8
	VSHUFB	V2, V8, V8, V8	// byte-swap each 32-bit word
	VXORV	V8, V0, V0
	ADDV	$16, R5

	// R8 = nr - 1 (middle rounds with InvMixColumns)
	ADDV	$-1, R4, R8

Ldec_loop:
	SUBBYTES(V0, R10, V13)
	VSHUFB	V7, V13, V13, V0	// InvShiftRows

	INVMIXCOLUMNS(V0, V0, V4, V5, V6, V9, V10, V11, V12, V13, V14, V16)

	VMOVQ	(R5), V8		// AddRoundKey(put in last)
	VSHUFB	V2, V8, V8, V8
	VXORV	V8, V0, V0
	ADDV	$16, R5

	SUBV	$1, R8
	BNE	R8, R0, Ldec_loop

	// Final round: InvSubBytes + InvShiftRows + AddRoundKey (no InvMixColumns)
	SUBBYTES(V0, R10, V13)
	VSHUFB	V7, V13, V13, V0
	VMOVQ	(R5), V8
	VSHUFB	V2, V8, V8, V8
	VXORV	V8, V0, V0

	// Store result
	VMOVQ	V0, (R6)
	RET

// -----------------------------------------------------------------------
// func expandKeyAsm(nr int, key *byte, enc *uint32, dec *uint32)
//
// Uses scalar S-box lookup (sbox0) for subw(); no VSHUFB needed here
// since key schedule processes 4 bytes at a time, not 16.
// -----------------------------------------------------------------------
TEXT ·expandKeyAsm(SB), NOSPLIT, $0-32
	MOVV	nr+0(FP), R4
	MOVV	key+8(FP), R5
	MOVV	enc+16(FP), R6
	MOVV	dec+24(FP), R7

	MOVV	$sbox0_0(SB), R8	// S-box for subw()
	MOVV	$·powx(SB), R14		// Rcon table

	// roundKeysSize = (nr+1)*4
	ADDV	$1, R4, R9
	SLLV	$2, R9, R9		// R9 = (nr+1)*4

	// nk = nr - 6  (AES-128→4, AES-192→6, AES-256→8)
	ADDV	$-6, R4, R10

	// Precompute the "nk > 6" flag once, instead of re-deriving it
	// (via SUBV $6, R10, R18 + BEQ/BLT) on every iteration where pos==4.
	MOVV	$0, R25			// R25 = 0 (false) by default
	MOVV	$6, R18
	BGE	R18, R10, Lnk6_done	// nk <= 6  ->  flag stays false
	MOVV	$1, R25			// nk > 6   ->  flag = true

Lnk6_done:
	// Hoist the "pos == 4" comparison constant out of the per-word loop.
	MOVV	$4, R20

	MOVV	R6, R17			// save enc base for Phase 3

	// Phase 1: load initial key words (i = 0..nk-1)
	MOVV	$0, R11			// i = 0
Linit_loop:
	BEQ	R11, R10, Linit_done
	SLLV	$2, R11, R13
	ADDV	R5, R13, R13		// &key[i*4]
	MOVWU	(R13), R12		// load 4 bytes (little-endian)
	REVB2W	R12, R12		// → big-endian word
	MOVW	R12, (R6)		// enc[i] = word
	ADDV	$4, R6
	ADDV	$1, R11
	JMP	Linit_loop

Linit_done:
	// Phase 2: key expansion (i = nk..roundKeysSize-1)
	// pos = i%nk (counter, reset to 0 when reaches nk)
	// rcon_idx = i/nk - 1 (incremented each time pos wraps)
	MOVV	$0, R15		// pos = 0  (nk%nk = 0)
	MOVV	$0, R16		// rcon_idx = 0

Lexpand_loop:
	BEQ	R11, R9, Lexpand_enc_done

	MOVWU	-4(R6), R12		// t = enc[i-1]

	BNE	R15, R0, Lcheck_nk6	// pos != 0 → skip i%nk==0 branch

	// i%nk == 0: t = subw(rotw(t)) ^ (powx[rcon_idx] << 24)
	ROTR	$24, R12, R12		// rotw: rotate-left-8 = rotate-right-24
	SUBW_INLINE(R12, R8, R13, R19)
	ADDV	R14, R16, R13		// &powx[rcon_idx]
	MOVBU	(R13), R13
	SLLV	$24, R13, R13
	XOR	R13, R12		// t ^= rcon
	ADDV	$1, R16			// rcon_idx++
	JMP	Ldo_xor

Lcheck_nk6:
	BNE	R15, R20, Ldo_xor	// pos != 4 -> skip (R20 preloaded with 4)
	BEQ	R25, R0, Ldo_xor	// nk <= 6 -> skip (R25 preloaded flag)
	// nk > 6 && pos == 4: t = subw(t)
	SUBW_INLINE(R12, R8, R13, R19)

Ldo_xor:
	SLLV	$2, R10, R13
	SUBV	R13, R6, R13		// &enc[i-nk]  (R6 - nk*4)
	MOVWU	(R13), R13		// enc[i-nk]
	XOR	R13, R12		// t ^= enc[i-nk]
	MOVW	R12, (R6)		// enc[i] = t
	ADDV	$4, R6
	ADDV	$1, R11
	ADDV	$1, R15			// pos++
	BNE	R15, R10, Lexpand_loop
	MOVV	$0, R15			// pos reset
	JMP	Lexpand_loop

Lexpand_enc_done:
	BEQ	R7, R0, Lexpand_done	// dec == nil → skip

	// ---- Phase 3 (vectorized): derive dec[] from enc[] ----
	// n = roundKeysSize (R9), enc base = R17 (saved earlier), dec base = R7
	// Load rotation tables (rot1/rot2/rot3) needed by INVMIXCOLUMNS.
	MOVV	$rot1(SB), R26
	MOVV	$rot2(SB), R27
	MOVV	$rot3(SB), R28
	MOVV	$byteSwap32(SB), R24
	VMOVQ	(R26), V4
	VMOVQ	(R27), V5
	VMOVQ	(R28), V6
	VMOVQ	(R24), V2

	MOVV	$sbox0_0(SB), R29	// sbox0 table base (for SUBBYTES)

	MOVV	$0, R19			// i = 0
Ldec_outer_vec:
	BEQ	R19, R9, Lexpand_done
	SUBV	R19, R9, R20
	ADDV	$-4, R20, R20		// R20 = ei = n - i - 4
	SLLV	$2, R20, R25
	ADDV	R17, R25, R25		// &enc[ei]  (R17 = saved enc base)
	VMOVQ	(R25), V0		// load 4 consecutive enc words as one 16-byte group

	// boundary: first (i==0) and last (i+4==n) groups get no InvMixColumns
	BEQ	R19, R0, Ldec_copy_vec
	ADDV	$4, R19, R20
	BEQ	R20, R9, Ldec_copy_vec

	VSHUFB	V2, V0, V0, V0		// word-storage order -> state order (V2 = byteSwap32 index, same table used in decryptBlockAsm)
	INVMIXCOLUMNS(V0, V0, V4, V5, V6, V9, V10, V11, V12, V13, V14, V16)
	VSHUFB	V2, V0, V0, V0		// state order -> word-storage order (so decryptBlockAsm's own byte-swap-before-use still works)

Ldec_copy_vec:
	SLLV	$2, R19, R25
	ADDV	R7, R25, R25		// &dec[i]
	VMOVQ	V0, (R25)

	ADDV	$4, R19
	JMP	Ldec_outer_vec

Lexpand_done:
	RET
