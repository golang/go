// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build zos && s390x && gc

#include "textflag.h"

//  provide the address of function variable to be fixed up.

TEXT ·getPipe2Addr(SB), NOSPLIT|NOFRAME, $0-8
	MOVD $·Pipe2(SB), R8
	MOVD R8, ret+0(FP)
	RET

TEXT ·get_FlockAddr(SB), NOSPLIT|NOFRAME, $0-8
	MOVD $·Flock(SB), R8
	MOVD R8, ret+0(FP)
	RET

TEXT ·get_GetxattrAddr(SB), NOSPLIT|NOFRAME, $0-8
	MOVD $·Getxattr(SB), R8
	MOVD R8, ret+0(FP)
	RET

TEXT ·get_NanosleepAddr(SB), NOSPLIT|NOFRAME, $0-8
	MOVD $·Nanosleep(SB), R8
	MOVD R8, ret+0(FP)
	RET

TEXT ·get_SetxattrAddr(SB), NOSPLIT|NOFRAME, $0-8
	MOVD $·Setxattr(SB), R8
	MOVD R8, ret+0(FP)
	RET

TEXT ·get_Wait4Addr(SB), NOSPLIT|NOFRAME, $0-8
	MOVD $·Wait4(SB), R8
	MOVD R8, ret+0(FP)
	RET

TEXT ·get_MountAddr(SB), NOSPLIT|NOFRAME, $0-8
	MOVD $·Mount(SB), R8
	MOVD R8, ret+0(FP)
	RET

TEXT ·get_UnmountAddr(SB), NOSPLIT|NOFRAME, $0-8
	MOVD $·Unmount(SB), R8
	MOVD R8, ret+0(FP)
	RET

TEXT ·get_UtimesNanoAtAddr(SB), NOSPLIT|NOFRAME, $0-8
	MOVD $·UtimesNanoAt(SB), R8
	MOVD R8, ret+0(FP)
	RET

TEXT ·get_UtimesNanoAddr(SB), NOSPLIT|NOFRAME, $0-8
	MOVD $·UtimesNano(SB), R8
	MOVD R8, ret+0(FP)
	RET

TEXT ·get_MkfifoatAddr(SB), NOSPLIT|NOFRAME, $0-8
	MOVD $·Mkfifoat(SB), R8
	MOVD R8, ret+0(FP)
	RET

TEXT ·get_ChtagAddr(SB), NOSPLIT|NOFRAME, $0-8
	MOVD $·Chtag(SB), R8
	MOVD R8, ret+0(FP)
	RET

TEXT ·get_ReadlinkatAddr(SB), NOSPLIT|NOFRAME, $0-8
	MOVD $·Readlinkat(SB), R8
	MOVD R8, ret+0(FP)
	RET
	
