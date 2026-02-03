// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "go_asm.h"
#include "textflag.h"

// Test-only.
TEXT ·ExpandAVX512(SB), NOSPLIT, $0-24
	MOVQ sizeClass+0(FP), CX
	MOVQ packed+8(FP), AX

	// Call the expander for this size class
	LEAQ ·gcExpandersAVX512(SB), BX
	CALL (BX)(CX*8)

	MOVQ unpacked+16(FP), DI // Expanded output bitmap pointer
	VMOVDQU64 Z1, 0(DI)
	VMOVDQU64 Z2, 64(DI)
	VZEROUPPER
	RET

TEXT ·scanSpanPackedAVX512(SB), NOSPLIT, $256-44
	// Z1+Z2 = Expand the grey object mask into a grey word mask
	MOVQ objMarks+16(FP), AX
	MOVQ sizeClass+24(FP), CX
	LEAQ ·gcExpandersAVX512(SB), BX
	CALL (BX)(CX*8)

	// Z3+Z4 = Load the pointer mask
	MOVQ ptrMask+32(FP), AX
	VMOVDQU64 0(AX), Z3
	VMOVDQU64 64(AX), Z4

	// Z1+Z2 = Combine the grey word mask with the pointer mask to get the scan mask
	VPANDQ Z1, Z3, Z1
	VPANDQ Z2, Z4, Z2

	// Now each bit of Z1+Z2 represents one word of the span.
	// Thus, each byte covers 64 bytes of memory, which is also how
	// much we can fix in a Z register.
	//
	// We do a load/compress for each 64 byte frame.
	//
	// Z3+Z4 [128]uint8 = Number of memory words to scan in each 64 byte frame
	VPOPCNTB Z1, Z3 // Requires BITALG
	VPOPCNTB Z2, Z4

	// Store the scan mask and word counts at 0(SP) and 128(SP).
	//
	// TODO: Is it better to read directly from the registers?
	VMOVDQU64 Z1, 0(SP)
	VMOVDQU64 Z2, 64(SP)
	VMOVDQU64 Z3, 128(SP)
	VMOVDQU64 Z4, 192(SP)

	// SI = Current address in span
	MOVQ mem+0(FP), SI
	// DI = Scan buffer base
	MOVQ bufp+8(FP), DI
	// DX = Index in scan buffer, (DI)(DX*8) = Current position in scan buffer
	MOVQ $0, DX

	// AX = address in scan mask, 128(AX) = address in popcount
	LEAQ 0(SP), AX

	// Loop over the 64 byte frames in this span.
	// BX = 1 past the end of the scan mask
	LEAQ 128(SP), BX

	// Align loop to a cache line so that performance is less sensitive
	// to how this function ends up laid out in memory. This is a hot
	// function in the GC, and this is a tight loop. We don't want
	// performance to waver wildly due to unrelated changes.
	PCALIGN $64
loop:
	// CX = Fetch the mask of words to load from this frame.
	MOVBQZX 0(AX), CX
	// Skip empty frames.
	TESTQ CX, CX
	JZ skip

	// Load the 64 byte frame.
	KMOVB CX, K1
	VMOVDQA64 0(SI), Z1

	// Collect just the pointers from the greyed objects into the scan buffer,
	// i.e., copy the word indices in the mask from Z1 into contiguous memory.
	//
	// N.B. VPCOMPRESSQ supports a memory destination. Unfortunately, on
	// AMD Genoa / Zen 4, using VPCOMPRESSQ with a memory destination
	// imposes a severe performance penalty of around an order of magnitude
	// compared to a register destination.
	//
	// This workaround is unfortunate on other microarchitectures, where a
	// memory destination is slightly faster than adding an additional move
	// instruction, but no where near an order of magnitude. It would be
	// nice to have a Genoa-only variant here.
	//
	// AMD Turin / Zen 5 fixes this issue.
	//
	// See
	// https://lemire.me/blog/2025/02/14/avx-512-gotcha-avoid-compressing-words-to-memory-with-amd-zen-4-processors/.
	VPCOMPRESSQ Z1, K1, Z2
	VMOVDQU64 Z2, (DI)(DX*8)

	// Advance the scan buffer position by the number of pointers.
	MOVBQZX 128(AX), CX
	ADDQ CX, DX

skip:
	ADDQ $64, SI
	ADDQ $1, AX
	CMPQ AX, BX
	JB loop

end:
	MOVL DX, count+40(FP)
	VZEROUPPER
	RET
