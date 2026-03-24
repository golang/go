// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Macros for transitioning from the host ABI to Go ABI0.
//
// These macros save and restore the callee-saved registers
// from the stack, but they don't adjust stack pointer, so
// the user should prepare stack space in advance.
// SAVE_GPR(offset) saves X8, X9, X18-X27 to the stack space
// of ((offset)+0*8)(X2) ~ ((offset)+11*8)(X2).
//
// SAVE_FPR(offset) saves F8, F9, F18-F27 to the stack space
// of ((offset)+0*8)(X2) ~ ((offset)+11*8)(X2).
//
// Note: g is X27

#define SAVE_GPR(offset) \
	MOV X8, ((offset)+0*8)(X2)   \
	MOV X9, ((offset)+1*8)(X2)   \
	MOV X18, ((offset)+2*8)(X2)  \
	MOV X19, ((offset)+3*8)(X2)  \
	MOV X20, ((offset)+4*8)(X2)  \
	MOV X21, ((offset)+5*8)(X2)  \
	MOV X22, ((offset)+6*8)(X2)  \
	MOV X23, ((offset)+7*8)(X2)  \
	MOV X24, ((offset)+8*8)(X2)  \
	MOV X25, ((offset)+9*8)(X2)  \
	MOV X26, ((offset)+10*8)(X2) \
	MOV g, ((offset)+11*8)(X2)

#define RESTORE_GPR(offset) \
	MOV ((offset)+0*8)(X2), X8   \
	MOV ((offset)+1*8)(X2), X9   \
	MOV ((offset)+2*8)(X2), X18  \
	MOV ((offset)+3*8)(X2), X19  \
	MOV ((offset)+4*8)(X2), X20  \
	MOV ((offset)+5*8)(X2), X21  \
	MOV ((offset)+6*8)(X2), X22  \
	MOV ((offset)+7*8)(X2), X23  \
	MOV ((offset)+8*8)(X2), X24  \
	MOV ((offset)+9*8)(X2), X25  \
	MOV ((offset)+10*8)(X2), X26 \
	MOV ((offset)+11*8)(X2), g

#define SAVE_FPR(offset) \
	MOVD F8, ((offset)+0*8)(X2)   \
	MOVD F9, ((offset)+1*8)(X2)   \
	MOVD F18, ((offset)+2*8)(X2)  \
	MOVD F19, ((offset)+3*8)(X2)  \
	MOVD F20, ((offset)+4*8)(X2)  \
	MOVD F21, ((offset)+5*8)(X2)  \
	MOVD F22, ((offset)+6*8)(X2)  \
	MOVD F23, ((offset)+7*8)(X2)  \
	MOVD F24, ((offset)+8*8)(X2)  \
	MOVD F25, ((offset)+9*8)(X2)  \
	MOVD F26, ((offset)+10*8)(X2) \
	MOVD F27, ((offset)+11*8)(X2)

#define RESTORE_FPR(offset) \
	MOVD ((offset)+0*8)(X2), F8   \
	MOVD ((offset)+1*8)(X2), F9   \
	MOVD ((offset)+2*8)(X2), F18  \
	MOVD ((offset)+3*8)(X2), F19  \
	MOVD ((offset)+4*8)(X2), F20  \
	MOVD ((offset)+5*8)(X2), F21  \
	MOVD ((offset)+6*8)(X2), F22  \
	MOVD ((offset)+7*8)(X2), F23  \
	MOVD ((offset)+8*8)(X2), F24  \
	MOVD ((offset)+9*8)(X2), F25  \
	MOVD ((offset)+10*8)(X2), F26 \
	MOVD ((offset)+11*8)(X2), F27
