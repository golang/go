// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

TEXT errors(SB),$0
	MOVL	foo<>(SB)(AX), AX	// ERROR "invalid instruction"
	MOVL	(AX)(SP*1), AX		// ERROR "invalid instruction"
	EXTRACTPS $4, X2, (BX)          // ERROR "invalid instruction"
	EXTRACTPS $-1, X2, (BX)         // ERROR "invalid instruction"
	// VSIB addressing does not permit non-vector (X/Y)
	// scaled index register.
	VPGATHERDQ X12,(R13)(AX*2), X11 // ERROR "invalid instruction"
	VPGATHERDQ X2, 664(BX*1), X1    // ERROR "invalid instruction"
	VPGATHERDQ Y2, (BP)(AX*2), Y1   // ERROR "invalid instruction"
	VPGATHERDQ Y5, 664(DX*8), Y6    // ERROR "invalid instruction"
	VPGATHERDQ Y5, (DX), Y0         // ERROR "invalid instruction"
	// VM/X rejects Y index register.
	VPGATHERDQ Y5, 664(Y14*8), Y6   // ERROR "invalid instruction"
	VPGATHERQQ X2, (BP)(Y7*2), X1   // ERROR "invalid instruction"
	// VM/Y rejects X index register.
	VPGATHERQQ Y2, (BP)(X7*2), Y1   // ERROR "invalid instruction"
	VPGATHERDD Y5, -8(X14*8), Y6    // ERROR "invalid instruction"
	// No VSIB for legacy instructions.
	MOVL (AX)(X0*1), AX             // ERROR "invalid instruction"
	MOVL (AX)(Y0*1), AX             // ERROR "invalid instruction"
	// VSIB/VM is invalid without vector index.
	// TODO(quasilyte): improve error message (#21860).
	// "invalid VSIB address (missing vector index)"
	VPGATHERQQ Y2, (BP), Y1         // ERROR "invalid instruction"
	// AVX2GATHER mask/index/dest #UD cases.
	VPGATHERQQ Y2, (BP)(X2*2), Y2   // ERROR "mask, index, and destination registers should be distinct"
	VPGATHERQQ Y2, (BP)(X2*2), Y7   // ERROR "mask, index, and destination registers should be distinct"
	VPGATHERQQ Y2, (BP)(X7*2), Y2   // ERROR "mask, index, and destination registers should be distinct"
	VPGATHERQQ Y7, (BP)(X2*2), Y2   // ERROR "mask, index, and destination registers should be distinct"
	VPGATHERDQ X2, 664(X2*8), X2    // ERROR "mask, index, and destination registers should be distinct"
	VPGATHERDQ X2, 664(X2*8), X7    // ERROR "mask, index, and destination registers should be distinct"
	VPGATHERDQ X2, 664(X7*8), X2    // ERROR "mask, index, and destination registers should be distinct"
	VPGATHERDQ X7, 664(X2*8), X2    // ERROR "mask, index, and destination registers should be distinct"
	// Non-X0 for Yxr0 should produce an error
	BLENDVPD X1, (BX), X2           // ERROR "invalid instruction"
	// Check offset overflow. Must fit in int32.
	MOVQ 2147483647+1(AX), AX       // ERROR "offset too large"
	MOVQ 3395469782(R10), R8        // ERROR "offset too large"
	LEAQ 3395469782(AX), AX         // ERROR "offset too large"
	ADDQ 3395469782(AX), AX         // ERROR "offset too large"
	ADDL 3395469782(AX), AX         // ERROR "offset too large"
	ADDW 3395469782(AX), AX         // ERROR "offset too large"
	LEAQ 433954697820(AX), AX       // ERROR "offset too large"
	ADDQ 433954697820(AX), AX       // ERROR "offset too large"
	ADDL 433954697820(AX), AX       // ERROR "offset too large"
	ADDW 433954697820(AX), AX       // ERROR "offset too large"
	// Pseudo-registers should not be used as scaled index.
	CALL (AX)(PC*1)                 // ERROR "invalid instruction"
	CALL (AX)(SB*1)                 // ERROR "invalid instruction"
	CALL (AX)(FP*1)                 // ERROR "invalid instruction"
	// Forbid memory operands for MOV CR/DR. See #24981.
	MOVQ CR0, (AX)                  // ERROR "invalid instruction"
	MOVQ CR2, (AX)                  // ERROR "invalid instruction"
	MOVQ CR3, (AX)                  // ERROR "invalid instruction"
	MOVQ CR4, (AX)                  // ERROR "invalid instruction"
	MOVQ CR8, (AX)                  // ERROR "invalid instruction"
	MOVQ (AX), CR0                  // ERROR "invalid instruction"
	MOVQ (AX), CR2                  // ERROR "invalid instruction"
	MOVQ (AX), CR3                  // ERROR "invalid instruction"
	MOVQ (AX), CR4                  // ERROR "invalid instruction"
	MOVQ (AX), CR8                  // ERROR "invalid instruction"
	MOVQ DR0, (AX)                  // ERROR "invalid instruction"
	MOVQ DR2, (AX)                  // ERROR "invalid instruction"
	MOVQ DR3, (AX)                  // ERROR "invalid instruction"
	MOVQ DR6, (AX)                  // ERROR "invalid instruction"
	MOVQ DR7, (AX)                  // ERROR "invalid instruction"
	MOVQ (AX), DR0                  // ERROR "invalid instruction"
	MOVQ (AX), DR2                  // ERROR "invalid instruction"
	MOVQ (AX), DR3                  // ERROR "invalid instruction"
	MOVQ (AX), DR6                  // ERROR "invalid instruction"
	MOVQ (AX), DR7                  // ERROR "invalid instruction"
	// AVX512GATHER index/index #UD cases.
	VPGATHERQQ (BP)(X2*2), K1, X2   // ERROR "index and destination registers should be distinct"
	VPGATHERQQ (BP)(Y15*2), K1, Y15 // ERROR "index and destination registers should be distinct"
	VPGATHERQQ (BP)(Z20*2), K1, Z20 // ERROR "index and destination registers should be distinct"
	VPGATHERDQ (BP)(X2*2), K1, X2   // ERROR "index and destination registers should be distinct"
	VPGATHERDQ (BP)(X15*2), K1, Y15 // ERROR "index and destination registers should be distinct"
	VPGATHERDQ (BP)(Y20*2), K1, Z20 // ERROR "index and destination registers should be distinct"
	// Instructions without EVEX variant can't use High-16 registers.
	VADDSUBPD X20, X1, X2           // ERROR "invalid instruction"
	VADDSUBPS X0, X20, X2           // ERROR "invalid instruction"
	// Use of K0 for write mask (Yknot0).
	// TODO(quasilyte): improve error message (#21860).
	//                  "K0 can't be used for write mask"
	VADDPD X0, X1, K0, X2           // ERROR "invalid instruction"
	VADDPD Y0, Y1, K0, Y2           // ERROR "invalid instruction"
	VADDPD Z0, Z1, K0, Z2           // ERROR "invalid instruction"
	// VEX-encoded VSIB can't use High-16 registers as index (unlike EVEX).
	// TODO(quasilyte): improve error message (#21860).
	VPGATHERQQ X2, (BP)(X20*2), X3  // ERROR "invalid instruction"
	VPGATHERQQ Y2, (BP)(Y20*2), Y3  // ERROR "invalid instruction"
	// YzrMulti4 expects exactly 4 registers referenced by REG_LIST.
	// TODO(quasilyte): improve error message (#21860).
	V4FMADDPS (AX), [Z0-Z4], K1, Z7  // ERROR "invalid instruction"
	V4FMADDPS (AX), [Z0-Z0], K1, Z7  // ERROR "invalid instruction"
	// Invalid ranges in REG_LIST (low > high).
	// TODO(quasilyte): improve error message (#21860).
	V4FMADDPS (AX), [Z4-Z0], K1, Z7  // ERROR "invalid instruction"
	V4FMADDPS (AX), [Z1-Z0], K1, Z7  // ERROR "invalid instruction"
	// Mismatching registers in a range.
	// TODO(quasilyte): improve error message (#21860).
	V4FMADDPS (AX), [AX-Z3], K1, Z7  // ERROR "invalid instruction"
	V4FMADDPS (AX), [Z0-AX], K1, Z7  // ERROR "invalid instruction"
	// Usage of suffixes for non-EVEX instructions.
	ADCB.Z $7, AL                    // ERROR "invalid instruction"
	ADCB.RU_SAE $7, AL               // ERROR "invalid instruction"
	ADCB.RU_SAE.Z $7, AL             // ERROR "invalid instruction"
	// Usage of rounding with invalid operands.
	VADDPD.RU_SAE X3, X2, K1, X1     // ERROR "unsupported rounding"
	VADDPD.RD_SAE X3, X2, K1, X1     // ERROR "unsupported rounding"
	VADDPD.RZ_SAE X3, X2, K1, X1     // ERROR "unsupported rounding"
	VADDPD.RN_SAE X3, X2, K1, X1     // ERROR "unsupported rounding"
	VADDPD.RU_SAE Y3, Y2, K1, Y1     // ERROR "unsupported rounding"
	VADDPD.RD_SAE Y3, Y2, K1, Y1     // ERROR "unsupported rounding"
	VADDPD.RZ_SAE Y3, Y2, K1, Y1     // ERROR "unsupported rounding"
	VADDPD.RN_SAE Y3, Y2, K1, Y1     // ERROR "unsupported rounding"
	// Unsupported SAE.
	VMAXPD.SAE (AX), Z2, K1, Z1      // ERROR "illegal SAE with memory argument"
	VADDPD.SAE X3, X2, K1, X1        // ERROR "unsupported SAE"
	// Unsupported zeroing.
	VFPCLASSPDX.Z $0, (AX), K2, K1   // ERROR "unsupported zeroing"
	VFPCLASSPDY.Z $0, (AX), K2, K1   // ERROR "unsupported zeroing"
	// Unsupported broadcast.
	VFPCLASSSD.BCST $0, (AX), K2, K1 // ERROR "unsupported broadcast"
	VFPCLASSSS.BCST $0, (AX), K2, K1 // ERROR "unsupported broadcast"
	// Broadcast without memory operand.
	VADDPD.BCST X3, X2, K1, X1       // ERROR "illegal broadcast without memory argument"
	VADDPD.BCST X3, X2, K1, X1       // ERROR "illegal broadcast without memory argument"
	VADDPD.BCST X3, X2, K1, X1       // ERROR "illegal broadcast without memory argument"
	// CLWB instructions:
	CLWB BX                          // ERROR "invalid instruction"
	// CLDEMOTE instructions:
	CLDEMOTE BX                      // ERROR "invalid instruction"
	// WAITPKG instructions:
	TPAUSE (BX)                      // ERROR "invalid instruction"
	UMONITOR (BX)                    // ERROR "invalid instruction"
	UMWAIT (BX)                      // ERROR "invalid instruction"
	// .Z instructions
	VMOVDQA32.Z Z0, Z1               // ERROR "mask register must be specified for .Z instructions"
	VMOVDQA32.Z Z0, K0, Z1           // ERROR "invalid instruction"
	VMOVDQA32.Z Z0, K1, Z1           // ok

	RDPID (BX)			 // ERROR "invalid instruction"

	RET
