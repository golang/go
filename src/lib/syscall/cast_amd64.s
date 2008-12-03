// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// conversion operators - really just casts
TEXT	syscall·BytePtr(SB),7,$-8
	MOVQ	8(SP), AX
	MOVQ	AX, 16(SP)
	RET

TEXT	syscall·BytePtrPtr(SB),7,$-8
	MOVQ	8(SP), AX
	MOVQ	AX, 16(SP)
	RET

TEXT	syscall·Int32Ptr(SB),7,$-8
	MOVQ	8(SP), AX
	MOVQ	AX, 16(SP)
	RET

TEXT	syscall·Int64Ptr(SB),7,$-8
	MOVQ	8(SP), AX
	MOVQ	AX, 16(SP)
	RET

TEXT	syscall·KeventPtr(SB),7,$-8
	MOVQ	8(SP), AX
	MOVQ	AX, 16(SP)
	RET

TEXT	syscall·EpollEventPtr(SB),7,$-8
	MOVQ	8(SP), AX
	MOVQ	AX, 16(SP)
	RET

TEXT	syscall·LingerPtr(SB),7,$-8
	MOVQ	8(SP), AX
	MOVQ	AX, 16(SP)
	RET

TEXT	syscall·SockaddrPtr(SB),7,$-8
	MOVQ	8(SP), AX
	MOVQ	AX, 16(SP)
	RET

TEXT	syscall·StatPtr(SB),7,$-8
	MOVQ	8(SP), AX
	MOVQ	AX, 16(SP)
	RET

TEXT	syscall·TimespecPtr(SB),7,$-8
	MOVQ	8(SP), AX
	MOVQ	AX, 16(SP)
	RET

TEXT	syscall·TimevalPtr(SB),7,$-8
	MOVQ	8(SP), AX
	MOVQ	AX, 16(SP)
	RET

TEXT	syscall·RusagePtr(SB),7,$-8
	MOVQ	8(SP), AX
	MOVQ	AX, 16(SP)
	RET

TEXT	syscall·SockaddrToSockaddrInet4(SB),7,$-8
	MOVQ	8(SP), AX
	MOVQ	AX, 16(SP)
	RET

TEXT	syscall·SockaddrToSockaddrInet6(SB),7,$-8
	MOVQ	8(SP), AX
	MOVQ	AX, 16(SP)
	RET

TEXT	syscall·SockaddrInet4ToSockaddr(SB),7,$-8
	MOVQ	8(SP), AX
	MOVQ	AX, 16(SP)
	RET

TEXT	syscall·SockaddrInet6ToSockaddr(SB),7,$-8
	MOVQ	8(SP), AX
	MOVQ	AX, 16(SP)
	RET

