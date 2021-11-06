// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This is an implementation based on the s390x
// implementation.

// Find a separator with 2 <= len <= 32 within a string.
// Separators with lengths of 2, 3 or 4 are handled
// specially.

// This works on power8 and above. The loads and
// compares are done in big endian order
// since that allows the used of VCLZD, and allows
// the same implementation to work on big and little
// endian platforms with minimal conditional changes.

// NOTE: There is a power9 implementation that
// improves performance by 10-15% on little
// endian for some of the benchmarks, but
// work is still needed for a big endian
// implementation on power9.

//go:build ppc64 || ppc64le

#include "go_asm.h"
#include "textflag.h"

// Needed to swap LXVD2X loads to the correct
// byte order to work on POWER8.

#ifdef GOARCH_ppc64
DATA byteswap<>+0(SB)/8, $0x0001020304050607
DATA byteswap<>+8(SB)/8, $0x08090a0b0c0d0e0f
#else
DATA byteswap<>+0(SB)/8, $0x0706050403020100
DATA byteswap<>+8(SB)/8, $0x0f0e0d0c0b0a0908
#endif

// Load bytes in big endian order. Address
// alignment does not need checking.
#define VLOADSWAP(base, index, vreg, vsreg) \
	LXVD2X (base)(index), vsreg;  \
	VPERM  vreg, vreg, SWAP, vreg

GLOBL byteswap<>+0(SB), RODATA, $16

TEXT ·Index<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-56
#ifdef GOEXPERIMENT_regabiargs 
// R3 = byte array pointer 
// R4 = length 
        MOVD R6,R5             // R5 = separator pointer
        MOVD R7,R6             // R6 = separator length 
#else
	MOVD a_base+0(FP), R3  // R3 = byte array pointer
	MOVD a_len+8(FP), R4   // R4 = length
	MOVD b_base+24(FP), R5 // R5 = separator pointer
	MOVD b_len+32(FP), R6  // R6 = separator length
	MOVD $ret+48(FP), R14  // R14 = &ret
#endif


#ifdef GOARCH_ppc64le
	MOVBZ internal∕cpu·PPC64+const_offsetPPC64HasPOWER9(SB), R7
	CMP   R7, $1
	BNE   power8
	BR    indexbodyp9<>(SB)

#endif
power8:
	BR indexbody<>(SB)

TEXT ·IndexString<ABIInternal>(SB),NOSPLIT|NOFRAME,$0-40
#ifndef GOEXPERIMENT_regabiargs
	MOVD a_base+0(FP), R3  // R3 = string
	MOVD a_len+8(FP), R4   // R4 = length
	MOVD b_base+16(FP), R5 // R5 = separator pointer
	MOVD b_len+24(FP), R6  // R6 = separator length
	MOVD $ret+32(FP), R14  // R14 = &ret
#endif


#ifdef GOARCH_ppc64le
	MOVBZ internal∕cpu·PPC64+const_offsetPPC64HasPOWER9(SB), R7
	CMP   R7, $1
	BNE   power8
	BR    indexbodyp9<>(SB)

#endif
power8:
	BR indexbody<>(SB)

	// s: string we are searching
	// sep: string to search for
	// R3=&s[0], R4=len(s)
	// R5=&sep[0], R6=len(sep)
	// R14=&ret (index where sep found)
	// R7=working addr of string
	// R16=index value 16
	// R17=index value 17
	// R18=index value 18
	// R19=index value 1
	// R26=LASTBYTE of string
	// R27=LASTSTR last start byte to compare with sep
	// R8, R9 scratch
	// V0=sep left justified zero fill
	// CR4=sep length >= 16

#define SEPMASK V17
#define LASTBYTE R26
#define LASTSTR R27
#define ONES V20
#define SWAP V21
#define V0_ VS32
#define V1_ VS33
#define V2_ VS34
#define V3_ VS35
#define V4_ VS36
#define V5_ VS37
#define V6_ VS38
#define V7_ VS39
#define V8_ VS40
#define V9_ VS41
#define SWAP_ VS53
TEXT indexbody<>(SB), NOSPLIT|NOFRAME, $0
	CMP      R6, R4                 // Compare lengths
	BGT      notfound               // If sep len is > string, notfound
	ADD      R4, R3, LASTBYTE       // find last byte addr
	SUB      R6, LASTBYTE, LASTSTR  // LAST=&s[len(s)-len(sep)] (last valid start index)
	CMP      R6, $0                 // Check sep len
	BEQ      notfound               // sep len 0 -- not found
	MOVD     R3, R7                 // Copy of string addr
	MOVD     $16, R16               // Index value 16
	MOVD     $17, R17               // Index value 17
	MOVD     $18, R18               // Index value 18
	MOVD     $1, R19                // Index value 1
	MOVD     $byteswap<>+00(SB), R8
	VSPLTISB $0xFF, ONES            // splat all 1s
	LXVD2X   (R8)(R0), SWAP_        // Set up swap string

	CMP    R6, $16, CR4        // CR4 for len(sep) >= 16
	VOR    ONES, ONES, SEPMASK // Set up full SEPMASK
	BGE    CR4, loadge16       // Load for len(sep) >= 16
	SUB    R6, R16, R9         // 16-len of sep
	SLD    $3, R9              // Set up for VSLO
	MTVSRD R9, V9_             // Set up for VSLO
	VSLDOI $8, V9, V9, V9      // Set up for VSLO
	VSLO   ONES, V9, SEPMASK   // Mask for separator len(sep) < 16

loadge16:
	ANDCC $15, R5, R9 // Find byte offset of sep
	ADD   R9, R6, R10 // Add sep len
	CMP   R10, $16    // Check if sep len+offset > 16
	BGT   sepcross16  // Sep crosses 16 byte boundary

	RLDICR $0, R5, $59, R8 // Adjust addr to 16 byte container
	VLOADSWAP(R8, R0, V0, V0_)// Load 16 bytes @R8 into V0
	SLD    $3, R9          // Set up shift count for VSLO
	MTVSRD R9, V8_         // Set up shift count for VSLO
	VSLDOI $8, V8, V8, V8
	VSLO   V0, V8, V0      // Shift by start byte

	VAND V0, SEPMASK, V0 // Mask separator (< 16)
	BR   index2plus

sepcross16:
	VLOADSWAP(R5, R0, V0, V0_) // Load 16 bytes @R5 into V0

	VAND V0, SEPMASK, V0 // mask out separator
	BLE  CR4, index2to16
	BR   index17plus     // Handle sep > 16

index2plus:
	CMP      R6, $2       // Check length of sep
	BNE      index3plus   // If not 2, check for 3
	ADD      $16, R7, R9  // Check if next 16 bytes past last
	CMP      R9, LASTBYTE // compare with last
	BGE      index2to16   // 2 <= len(string) <= 16
	MOVD     $0xff00, R21 // Mask for later
	MTVSRD   R21, V25     // Move to Vreg
	VSPLTH   $3, V25, V31 // Splat mask
	VSPLTH   $0, V0, V1   // Splat 1st 2 bytes of sep
	VSPLTISB $0, V10      // Clear V10

	// First case: 2 byte separator
	// V1: 2 byte separator splatted
	// V2: 16 bytes at addr
	// V4: 16 bytes at addr+1
	// Compare 2 byte separator at start
	// and at start+1. Use VSEL to combine
	// those results to find the first
	// matching start byte, returning
	// that value when found. Loop as
	// long as len(string) > 16
index2loop2:
	VLOADSWAP(R7, R19, V3, V3_) // Load 16 bytes @R7+1 into V3

index2loop:
	VLOADSWAP(R7, R0, V2, V2_) // Load 16 bytes @R7 into V2
	VCMPEQUH V1, V2, V5        // Search for sep
	VCMPEQUH V1, V3, V6        // Search for sep offset by 1
	VSEL     V6, V5, V31, V7   // merge even and odd indices
	VCLZD    V7, V18           // find index of first match
	MFVSRD   V18, R25          // get first value
	CMP      R25, $64          // Found if < 64
	BLT      foundR25          // Return byte index where found
	VSLDOI   $8, V18, V18, V18 // Adjust 2nd value
	MFVSRD   V18, R25          // get second value
	CMP      R25, $64          // Found if < 64
	ADD      $64, R25          // Update byte offset
	BLT      foundR25          // Return value
	ADD      $16, R7           // R7+=16 Update string pointer
	ADD      $17, R7, R9       // R9=F7+17 since loop unrolled
	CMP      R9, LASTBYTE      // Compare addr+17 against last byte
	BLT      index2loop2       // If < last, continue loop
	CMP      R7, LASTBYTE      // Compare addr+16 against last byte
	BLT      index2to16        // If < 16 handle specially
	VLOADSWAP(R7, R0, V3, V3_) // Load 16 bytes @R7 into V3
	VSLDOI   $1, V3, V10, V3   // Shift left by 1 byte
	BR       index2loop

index3plus:
	CMP    R6, $3       // Check if sep == 3
	BNE    index4plus   // If not check larger
	ADD    $19, R7, R9  // Find bytes for use in this loop
	CMP    R9, LASTBYTE // Compare against last byte
	BGE    index2to16   // Remaining string 2<=len<=16
	MOVD   $0xff00, R21 // Set up mask for upcoming loop
	MTVSRD R21, V25     // Move mask to Vreg
	VSPLTH $3, V25, V31 // Splat mask
	VSPLTH $0, V0, V1   // Splat 1st two bytes of sep
	VSPLTB $2, V0, V8   // Splat 3rd byte of sep

	// Loop to process 3 byte separator.
	// string[0:16] is in V2
	// string[2:18] is in V3
	// sep[0:2] splatted in V1
	// sec[3] splatted in v8
	// Load vectors at string, string+1
	// and string+2. Compare string, string+1
	// against first 2 bytes of separator
	// splatted, and string+2 against 3rd
	// byte splatted. Merge the results with
	// VSEL to find the first byte of a match.

	// Special handling for last 16 bytes if the
	// string fits in 16 byte multiple.
index3loop2:
	MOVD     $2, R21          // Set up index for 2
	VSPLTISB $0, V10          // Clear V10
	VLOADSWAP(R7, R21, V3, V3_)// Load 16 bytes @R7+2 into V3
	VSLDOI   $14, V3, V10, V3 // Left justify next 2 bytes

index3loop:
	VLOADSWAP(R7, R0, V2, V2_) // Load with correct order
	VSLDOI   $1, V2, V3, V4    // string[1:17]
	VSLDOI   $2, V2, V3, V9    // string[2:18]
	VCMPEQUH V1, V2, V5        // compare hw even indices
	VCMPEQUH V1, V4, V6        // compare hw odd indices
	VCMPEQUB V8, V9, V10       // compare 3rd to last byte
	VSEL     V6, V5, V31, V7   // Find 1st matching byte using mask
	VAND     V7, V10, V7       // AND matched bytes with matched 3rd byte
	VCLZD    V7, V18           // Find first nonzero indexes
	MFVSRD   V18, R25          // Move 1st doubleword
	CMP      R25, $64          // If < 64 found
	BLT      foundR25          // Return matching index
	VSLDOI   $8, V18, V18, V18 // Move value
	MFVSRD   V18, R25          // Move 2nd doubleword
	CMP      R25, $64          // If < 64 found
	ADD      $64, R25          // Update byte index
	BLT      foundR25          // Return matching index
	ADD      $16, R7           // R7+=16 string ptr
	ADD      $19, R7, R9       // Number of string bytes for loop
	CMP      R9, LASTBYTE      // Compare against last byte of string
	BLT      index3loop2       // If within, continue this loop
	CMP      R7, LASTSTR       // Compare against last start byte
	BLT      index2to16        // Process remainder
	VSPLTISB $0, V3            // Special case for last 16 bytes
	BR       index3loop        // Continue this loop

	// Loop to process 4 byte separator
	// string[0:16] in V2
	// string[3:16] in V3
	// sep[0:4] splatted in V1
	// Set up vectors with strings at offsets
	// 0, 1, 2, 3 and compare against the 4 byte
	// separator also splatted. Use VSEL with the
	// compare results to find the first byte where
	// a separator match is found.
index4plus:
	CMP  R6, $4       // Check if 4 byte separator
	BNE  index5plus   // If not next higher
	ADD  $20, R7, R9  // Check string size to load
	CMP  R9, LASTBYTE // Verify string length
	BGE  index2to16   // If not large enough, process remaining
	MOVD $2, R15      // Set up index

	// Set up masks for use with VSEL
	MOVD   $0xff, R21        // Set up mask 0xff000000ff000000...
	SLD    $24, R21
	MTVSRD R21, V10
	VSPLTW $1, V10, V29
	VSLDOI $2, V29, V29, V30 // Mask 0x0000ff000000ff00...
	MOVD   $0xffff, R21
	SLD    $16, R21
	MTVSRD R21, V10
	VSPLTW $1, V10, V31      // Mask 0xffff0000ffff0000...
	VSPLTW $0, V0, V1        // Splat 1st word of separator

index4loop:
	VLOADSWAP(R7, R0, V2, V2_) // Load 16 bytes @R7 into V2

next4:
	VSPLTISB $0, V10            // Clear
	MOVD     $3, R9             // Number of bytes beyond 16
	VLOADSWAP(R7, R9, V3, V3_)  // Load 16 bytes @R7+3 into V3
	VSLDOI   $13, V3, V10, V3   // Shift left last 3 bytes
	VSLDOI   $1, V2, V3, V4     // V4=(V2:V3)<<1
	VSLDOI   $2, V2, V3, V9     // V9=(V2:V3)<<2
	VSLDOI   $3, V2, V3, V10    // V10=(V2:v3)<<3
	VCMPEQUW V1, V2, V5         // compare index 0, 4, ... with sep
	VCMPEQUW V1, V4, V6         // compare index 1, 5, ... with sep
	VCMPEQUW V1, V9, V11        // compare index 2, 6, ... with sep
	VCMPEQUW V1, V10, V12       // compare index 3, 7, ... with sep
	VSEL     V6, V5, V29, V13   // merge index 0, 1, 4, 5, using mask
	VSEL     V12, V11, V30, V14 // merge index 2, 3, 6, 7, using mask
	VSEL     V14, V13, V31, V7  // final merge
	VCLZD    V7, V18            // Find first index for each half
	MFVSRD   V18, R25           // Isolate value
	CMP      R25, $64           // If < 64, found
	BLT      foundR25           // Return found index
	VSLDOI   $8, V18, V18, V18  // Move for MFVSRD
	MFVSRD   V18, R25           // Isolate other value
	CMP      R25, $64           // If < 64, found
	ADD      $64, R25           // Update index for high doubleword
	BLT      foundR25           // Return found index
	ADD      $16, R7            // R7+=16 for next string
	ADD      $20, R7, R9        // R+20 for all bytes to load
	CMP      R9, LASTBYTE       // Past end? Maybe check for extra?
	BLT      index4loop         // If not, continue loop
	CMP      R7, LASTSTR        // Check remainder
	BLE      index2to16         // Process remainder
	BR       notfound           // Not found

index5plus:
	CMP R6, $16     // Check for sep > 16
	BGT index17plus // Handle large sep

	// Assumption is that the separator is smaller than the string at this point
index2to16:
	CMP R7, LASTSTR // Compare last start byte
	BGT notfound    // last takes len(sep) into account

	ADD $16, R7, R9    // Check for last byte of string
	CMP R9, LASTBYTE
	BGT index2to16tail

	// At least 16 bytes of string left
	// Mask the number of bytes in sep
index2to16loop:
	VLOADSWAP(R7, R0, V1, V1_) // Load 16 bytes @R7 into V1

compare:
	VAND       V1, SEPMASK, V2 // Mask out sep size
	VCMPEQUBCC V0, V2, V3      // Compare masked string
	BLT        CR6, found      // All equal
	ADD        $1, R7          // Update ptr to next byte
	CMP        R7, LASTSTR     // Still less than last start byte
	BGT        notfound        // Not found
	ADD        $16, R7, R9     // Verify remaining bytes
	CMP        R9, LASTBYTE    // At least 16
	BLT        index2to16loop  // Try again

	// Less than 16 bytes remaining in string
	// Separator >= 2
index2to16tail:
	ADD   R3, R4, R9     // End of string
	SUB   R7, R9, R9     // Number of bytes left
	ANDCC $15, R7, R10   // 16 byte offset
	ADD   R10, R9, R11   // offset + len
	CMP   R11, $16       // >= 16?
	BLE   short          // Does not cross 16 bytes
	VLOADSWAP(R7, R0, V1, V1_)// Load 16 bytes @R7 into V1
	BR    index2to16next // Continue on

short:
	RLDICR   $0, R7, $59, R9 // Adjust addr to 16 byte container
	VLOADSWAP(R9, R0, V1, V1_)// Load 16 bytes @R9 into V1
	SLD      $3, R10         // Set up shift
	MTVSRD   R10, V8_        // Set up shift
	VSLDOI   $8, V8, V8, V8
	VSLO     V1, V8, V1      // Shift by start byte
	VSPLTISB $0, V25         // Clear for later use

index2to16next:
	VAND       V1, SEPMASK, V2 // Just compare size of sep
	VCMPEQUBCC V0, V2, V3      // Compare sep and partial string
	BLT        CR6, found      // Found
	ADD        $1, R7          // Not found, try next partial string
	CMP        R7, LASTSTR     // Check for end of string
	BGT        notfound        // If at end, then not found
	VSLDOI     $1, V1, V25, V1 // Shift string left by 1 byte
	BR         index2to16next  // Check the next partial string

index17plus:
	CMP      R6, $32      // Check if 17 < len(sep) <= 32
	BGT      index33plus
	SUB      $16, R6, R9  // Extra > 16
	SLD      $56, R9, R10 // Shift to use in VSLO
	MTVSRD   R10, V9_     // Set up for VSLO
	VLOADSWAP(R5, R9, V1, V1_)// Load 16 bytes @R5+R9 into V1
	VSLO     V1, V9, V1   // Shift left
	VSPLTISB $0xff, V7    // Splat 1s
	VSPLTISB $0, V27      // Splat 0

index17to32loop:
	VLOADSWAP(R7, R0, V2, V2_) // Load 16 bytes @R7 into V2

next17:
	VLOADSWAP(R7, R9, V3, V3_) // Load 16 bytes @R7+R9 into V3
	VSLO       V3, V9, V3      // Shift left
	VCMPEQUB   V0, V2, V4      // Compare first 16 bytes
	VCMPEQUB   V1, V3, V5      // Compare extra over 16 bytes
	VAND       V4, V5, V6      // Check if both equal
	VCMPEQUBCC V6, V7, V8      // All equal?
	BLT        CR6, found      // Yes
	ADD        $1, R7          // On to next byte
	CMP        R7, LASTSTR     // Check if last start byte
	BGT        notfound        // If too high, not found
	BR         index17to32loop // Continue

notfound:
#ifdef GOEXPERIMENT_regabiargs
        MOVD $-1, R3   // Return -1 if not found
#else
	MOVD $-1, R8   // Return -1 if not found
	MOVD R8, (R14)
#endif
	RET

index33plus:
	MOVD $0, (R0) // Case not implemented
	RET           // Crash before return

foundR25:
	SRD  $3, R25   // Convert from bits to bytes
	ADD  R25, R7   // Add to current string address
	SUB  R3, R7    // Subtract from start of string
#ifdef GOEXPERIMENT_regabiargs
        MOVD R7, R3    // Return byte where found
#else
	MOVD R7, (R14) // Return byte where found
#endif
	RET

found:
	SUB  R3, R7    // Return byte where found
#ifdef GOEXPERIMENT_regabiargs
        MOVD R7, R3
#else
	MOVD R7, (R14)
#endif
	RET

TEXT indexbodyp9<>(SB), NOSPLIT|NOFRAME, $0
	CMP      R6, R4                // Compare lengths
	BGT      notfound              // If sep len is > string, notfound
	ADD      R4, R3, LASTBYTE      // find last byte addr
	SUB      R6, LASTBYTE, LASTSTR // LAST=&s[len(s)-len(sep)] (last valid start index)
	CMP      R6, $0                // Check sep len
	BEQ      notfound              // sep len 0 -- not found
	MOVD     R3, R7                // Copy of string addr
	MOVD     $16, R16              // Index value 16
	MOVD     $17, R17              // Index value 17
	MOVD     $18, R18              // Index value 18
	MOVD     $1, R19               // Index value 1
	VSPLTISB $0xFF, ONES           // splat all 1s

	CMP    R6, $16, CR4        // CR4 for len(sep) >= 16
	VOR    ONES, ONES, SEPMASK // Set up full SEPMASK
	BGE    CR4, loadge16       // Load for len(sep) >= 16
	SUB    R6, R16, R9         // 16-len of sep
	SLD    $3, R9              // Set up for VSLO
	MTVSRD R9, V9_             // Set up for VSLO
	VSLDOI $8, V9, V9, V9      // Set up for VSLO
	VSLO   ONES, V9, SEPMASK   // Mask for separator len(sep) < 16

loadge16:
	ANDCC $15, R5, R9 // Find byte offset of sep
	ADD   R9, R6, R10 // Add sep len
	CMP   R10, $16    // Check if sep len+offset > 16
	BGT   sepcross16  // Sep crosses 16 byte boundary

	RLDICR  $0, R5, $59, R8 // Adjust addr to 16 byte container
	LXVB16X (R8)(R0), V0_   // Load 16 bytes @R8 into V0
	SLD     $3, R9          // Set up shift count for VSLO
	MTVSRD  R9, V8_         // Set up shift count for VSLO
	VSLDOI  $8, V8, V8, V8
	VSLO    V0, V8, V0      // Shift by start byte

	VAND V0, SEPMASK, V0 // Mask separator (< 16)
	BR   index2plus

sepcross16:
	LXVB16X (R5)(R0), V0_ // Load 16 bytes @R5 into V0

	VAND V0, SEPMASK, V0 // mask out separator
	BLE  CR4, index2to16
	BR   index17plus     // Handle sep > 16

index2plus:
	CMP      R6, $2       // Check length of sep
	BNE      index3plus   // If not 2, check for 3
	ADD      $16, R7, R9  // Check if next 16 bytes past last
	CMP      R9, LASTBYTE // compare with last
	BGE      index2to16   // 2 <= len(string) <= 16
	MOVD     $0xff00, R21 // Mask for later
	MTVSRD   R21, V25     // Move to Vreg
	VSPLTH   $3, V25, V31 // Splat mask
	VSPLTH   $0, V0, V1   // Splat 1st 2 bytes of sep
	VSPLTISB $0, V10      // Clear V10

	// First case: 2 byte separator
	// V1: 2 byte separator splatted
	// V2: 16 bytes at addr
	// V4: 16 bytes at addr+1
	// Compare 2 byte separator at start
	// and at start+1. Use VSEL to combine
	// those results to find the first
	// matching start byte, returning
	// that value when found. Loop as
	// long as len(string) > 16
index2loop2:
	LXVB16X (R7)(R19), V3_ // Load 16 bytes @R7+1 into V3

index2loop:
	LXVB16X  (R7)(R0), V2_   // Load 16 bytes @R7 into V2
	VCMPEQUH V1, V2, V5      // Search for sep
	VCMPEQUH V1, V3, V6      // Search for sep offset by 1
	VSEL     V6, V5, V31, V7 // merge even and odd indices
	VCLZD    V7, V18         // find index of first match
	MFVSRD   V18, R25        // get first value
	CMP      R25, $64        // Found if < 64
	BLT      foundR25        // Return byte index where found

	MFVSRLD V18, R25        // get second value
	CMP     R25, $64        // Found if < 64
	ADD     $64, R25        // Update byte offset
	BLT     foundR25        // Return value
	ADD     $16, R7         // R7+=16 Update string pointer
	ADD     $17, R7, R9     // R9=F7+17 since loop unrolled
	CMP     R9, LASTBYTE    // Compare addr+17 against last byte
	BLT     index2loop2     // If < last, continue loop
	CMP     R7, LASTBYTE    // Compare addr+16 against last byte
	BLT     index2to16      // If < 16 handle specially
	LXVB16X (R7)(R0), V3_   // Load 16 bytes @R7 into V3
	VSLDOI  $1, V3, V10, V3 // Shift left by 1 byte
	BR      index2loop

index3plus:
	CMP    R6, $3       // Check if sep == 3
	BNE    index4plus   // If not check larger
	ADD    $19, R7, R9  // Find bytes for use in this loop
	CMP    R9, LASTBYTE // Compare against last byte
	BGE    index2to16   // Remaining string 2<=len<=16
	MOVD   $0xff00, R21 // Set up mask for upcoming loop
	MTVSRD R21, V25     // Move mask to Vreg
	VSPLTH $3, V25, V31 // Splat mask
	VSPLTH $0, V0, V1   // Splat 1st two bytes of sep
	VSPLTB $2, V0, V8   // Splat 3rd byte of sep

	// Loop to process 3 byte separator.
	// string[0:16] is in V2
	// string[2:18] is in V3
	// sep[0:2] splatted in V1
	// sec[3] splatted in v8
	// Load vectors at string, string+1
	// and string+2. Compare string, string+1
	// against first 2 bytes of separator
	// splatted, and string+2 against 3rd
	// byte splatted. Merge the results with
	// VSEL to find the first byte of a match.

	// Special handling for last 16 bytes if the
	// string fits in 16 byte multiple.
index3loop2:
	MOVD     $2, R21          // Set up index for 2
	VSPLTISB $0, V10          // Clear V10
	LXVB16X  (R7)(R21), V3_   // Load 16 bytes @R7+2 into V3
	VSLDOI   $14, V3, V10, V3 // Left justify next 2 bytes

index3loop:
	LXVB16X  (R7)(R0), V2_   // Load 16 bytes @R7
	VSLDOI   $1, V2, V3, V4  // string[1:17]
	VSLDOI   $2, V2, V3, V9  // string[2:18]
	VCMPEQUH V1, V2, V5      // compare hw even indices
	VCMPEQUH V1, V4, V6      // compare hw odd indices
	VCMPEQUB V8, V9, V10     // compare 3rd to last byte
	VSEL     V6, V5, V31, V7 // Find 1st matching byte using mask
	VAND     V7, V10, V7     // AND matched bytes with matched 3rd byte
	VCLZD    V7, V18         // Find first nonzero indexes
	MFVSRD   V18, R25        // Move 1st doubleword
	CMP      R25, $64        // If < 64 found
	BLT      foundR25        // Return matching index

	MFVSRLD  V18, R25     // Move 2nd doubleword
	CMP      R25, $64     // If < 64 found
	ADD      $64, R25     // Update byte index
	BLT      foundR25     // Return matching index
	ADD      $16, R7      // R7+=16 string ptr
	ADD      $19, R7, R9  // Number of string bytes for loop
	CMP      R9, LASTBYTE // Compare against last byte of string
	BLT      index3loop2  // If within, continue this loop
	CMP      R7, LASTSTR  // Compare against last start byte
	BLT      index2to16   // Process remainder
	VSPLTISB $0, V3       // Special case for last 16 bytes
	BR       index3loop   // Continue this loop

	// Loop to process 4 byte separator
	// string[0:16] in V2
	// string[3:16] in V3
	// sep[0:4] splatted in V1
	// Set up vectors with strings at offsets
	// 0, 1, 2, 3 and compare against the 4 byte
	// separator also splatted. Use VSEL with the
	// compare results to find the first byte where
	// a separator match is found.
index4plus:
	CMP  R6, $4       // Check if 4 byte separator
	BNE  index5plus   // If not next higher
	ADD  $20, R7, R9  // Check string size to load
	CMP  R9, LASTBYTE // Verify string length
	BGE  index2to16   // If not large enough, process remaining
	MOVD $2, R15      // Set up index

	// Set up masks for use with VSEL
	MOVD    $0xff, R21 // Set up mask 0xff000000ff000000...
	SLD     $24, R21
	MTVSRWS R21, V29

	VSLDOI  $2, V29, V29, V30 // Mask 0x0000ff000000ff00...
	MOVD    $0xffff, R21
	SLD     $16, R21
	MTVSRWS R21, V31

	VSPLTW $0, V0, V1 // Splat 1st word of separator

index4loop:
	LXVB16X (R7)(R0), V2_ // Load 16 bytes @R7 into V2

next4:
	VSPLTISB $0, V10            // Clear
	MOVD     $3, R9             // Number of bytes beyond 16
	LXVB16X  (R7)(R9), V3_      // Load 16 bytes @R7 into V2
	VSLDOI   $13, V3, V10, V3   // Shift left last 3 bytes
	VSLDOI   $1, V2, V3, V4     // V4=(V2:V3)<<1
	VSLDOI   $2, V2, V3, V9     // V9=(V2:V3)<<2
	VSLDOI   $3, V2, V3, V10    // V10=(V2:v3)<<3
	VCMPEQUW V1, V2, V5         // compare index 0, 4, ... with sep
	VCMPEQUW V1, V4, V6         // compare index 1, 5, ... with sep
	VCMPEQUW V1, V9, V11        // compare index 2, 6, ... with sep
	VCMPEQUW V1, V10, V12       // compare index 3, 7, ... with sep
	VSEL     V6, V5, V29, V13   // merge index 0, 1, 4, 5, using mask
	VSEL     V12, V11, V30, V14 // merge index 2, 3, 6, 7, using mask
	VSEL     V14, V13, V31, V7  // final merge
	VCLZD    V7, V18            // Find first index for each half
	MFVSRD   V18, R25           // Isolate value
	CMP      R25, $64           // If < 64, found
	BLT      foundR25           // Return found index

	MFVSRLD V18, R25     // Isolate other value
	CMP     R25, $64     // If < 64, found
	ADD     $64, R25     // Update index for high doubleword
	BLT     foundR25     // Return found index
	ADD     $16, R7      // R7+=16 for next string
	ADD     $20, R7, R9  // R+20 for all bytes to load
	CMP     R9, LASTBYTE // Past end? Maybe check for extra?
	BLT     index4loop   // If not, continue loop
	CMP     R7, LASTSTR  // Check remainder
	BLE     index2to16   // Process remainder
	BR      notfound     // Not found

index5plus:
	CMP R6, $16     // Check for sep > 16
	BGT index17plus // Handle large sep

	// Assumption is that the separator is smaller than the string at this point
index2to16:
	CMP R7, LASTSTR // Compare last start byte
	BGT notfound    // last takes len(sep) into account

	ADD $16, R7, R9    // Check for last byte of string
	CMP R9, LASTBYTE
	BGT index2to16tail

	// At least 16 bytes of string left
	// Mask the number of bytes in sep
index2to16loop:
	LXVB16X (R7)(R0), V1_ // Load 16 bytes @R7 into V1

compare:
	VAND       V1, SEPMASK, V2 // Mask out sep size
	VCMPEQUBCC V0, V2, V3      // Compare masked string
	BLT        CR6, found      // All equal
	ADD        $1, R7          // Update ptr to next byte
	CMP        R7, LASTSTR     // Still less than last start byte
	BGT        notfound        // Not found
	ADD        $16, R7, R9     // Verify remaining bytes
	CMP        R9, LASTBYTE    // At least 16
	BLT        index2to16loop  // Try again

	// Less than 16 bytes remaining in string
	// Separator >= 2
index2to16tail:
	ADD     R3, R4, R9     // End of string
	SUB     R7, R9, R9     // Number of bytes left
	ANDCC   $15, R7, R10   // 16 byte offset
	ADD     R10, R9, R11   // offset + len
	CMP     R11, $16       // >= 16?
	BLE     short          // Does not cross 16 bytes
	LXVB16X (R7)(R0), V1_  // Load 16 bytes @R7 into V1
	BR      index2to16next // Continue on

short:
	RLDICR   $0, R7, $59, R9 // Adjust addr to 16 byte container
	LXVB16X  (R9)(R0), V1_   // Load 16 bytes @R9 into V1
	SLD      $3, R10         // Set up shift
	MTVSRD   R10, V8_        // Set up shift
	VSLDOI   $8, V8, V8, V8
	VSLO     V1, V8, V1      // Shift by start byte
	VSPLTISB $0, V25         // Clear for later use

index2to16next:
	VAND       V1, SEPMASK, V2 // Just compare size of sep
	VCMPEQUBCC V0, V2, V3      // Compare sep and partial string
	BLT        CR6, found      // Found
	ADD        $1, R7          // Not found, try next partial string
	CMP        R7, LASTSTR     // Check for end of string
	BGT        notfound        // If at end, then not found
	VSLDOI     $1, V1, V25, V1 // Shift string left by 1 byte
	BR         index2to16next  // Check the next partial string

index17plus:
	CMP      R6, $32       // Check if 17 < len(sep) <= 32
	BGT      index33plus
	SUB      $16, R6, R9   // Extra > 16
	SLD      $56, R9, R10  // Shift to use in VSLO
	MTVSRD   R10, V9_      // Set up for VSLO
	LXVB16X  (R5)(R9), V1_ // Load 16 bytes @R5+R9 into V1
	VSLO     V1, V9, V1    // Shift left
	VSPLTISB $0xff, V7     // Splat 1s
	VSPLTISB $0, V27       // Splat 0

index17to32loop:
	LXVB16X (R7)(R0), V2_ // Load 16 bytes @R7 into V2

next17:
	LXVB16X    (R7)(R9), V3_   // Load 16 bytes @R7+R9 into V3
	VSLO       V3, V9, V3      // Shift left
	VCMPEQUB   V0, V2, V4      // Compare first 16 bytes
	VCMPEQUB   V1, V3, V5      // Compare extra over 16 bytes
	VAND       V4, V5, V6      // Check if both equal
	VCMPEQUBCC V6, V7, V8      // All equal?
	BLT        CR6, found      // Yes
	ADD        $1, R7          // On to next byte
	CMP        R7, LASTSTR     // Check if last start byte
	BGT        notfound        // If too high, not found
	BR         index17to32loop // Continue

notfound:
#ifdef GOEXPERIMENT_regabiargs
        MOVD $-1, R3   // Return -1 if not found
#else
	MOVD $-1, R8   // Return -1 if not found
	MOVD R8, (R14)
#endif
	RET

index33plus:
	MOVD $0, (R0) // Case not implemented
	RET           // Crash before return

foundR25:
	SRD  $3, R25   // Convert from bits to bytes
	ADD  R25, R7   // Add to current string address
	SUB  R3, R7    // Subtract from start of string
#ifdef GOEXPERIMENT_regabiargs
        MOVD R7, R3    // Return byte where found
#else
	MOVD R7, (R14) // Return byte where found
#endif
	RET

found:
	SUB  R3, R7    // Return byte where found
#ifdef GOEXPERIMENT_regabiargs
        MOVD R7, R3
#else
	MOVD R7, (R14)
#endif
	RET

