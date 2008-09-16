// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Type-unsafe casts.

TEXT socket·SockaddrPtr(SB),7,$0
	MOVQ	8(SP), AX
	MOVQ	AX, 16(SP)
	RET
TEXT socket·Int32Ptr(SB),7,$0
	MOVQ	8(SP), AX
	MOVQ	AX, 16(SP)
	RET
TEXT socket·LingerPtr(SB),7,$0
	MOVQ	8(SP), AX
	MOVQ	AX, 16(SP)
	RET
TEXT	socket·TimevalPtr(SB),7,$0
	MOVQ	8(SP), AX
	MOVQ	AX, 16(SP)
	RET
TEXT socket·SockaddrInet4ToSockaddr(SB),7,$0
	MOVQ	8(SP), AX
	MOVQ	AX, 16(SP)
	RET
TEXT socket·SockaddrToSockaddrInet4(SB),7,$0
	MOVQ	8(SP), AX
	MOVQ	AX, 16(SP)
	RET
TEXT socket·SockaddrInet6ToSockaddr(SB),7,$0
	MOVQ	8(SP), AX
	MOVQ	AX, 16(SP)
	RET
TEXT socket·SockaddrToSockaddrInet6(SB),7,$0
	MOVQ	8(SP), AX
	MOVQ	AX, 16(SP)
	RET
