// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Registers
dst = 0
src = 1
n = 2
state = 3
pi = 4
pj = 5
i = 6
j = 7
k = 8
t = 11
t2 = 12

// func xorKeyStream(dst, src *byte, n int, state *[256]byte, i, j *uint8)
TEXT ·xorKeyStream(SB),7,$0
	MOVW 0(FP), R(dst)
	MOVW 4(FP), R(src)
	MOVW 8(FP), R(n)
	MOVW 12(FP), R(state)
	MOVW 16(FP), R(pi)
	MOVW 20(FP), R(pj)
	MOVBU (R(pi)), R(i)
	MOVBU (R(pj)), R(j)
	MOVW $0, R(k)

loop:
	// i += 1; j += state[i]
	ADD $1, R(i)
	AND $0xff, R(i)
	MOVBU R(i)<<2(R(state)), R(t)
	ADD R(t), R(j)
	AND $0xff, R(j)

	// swap state[i] <-> state[j]
	MOVBU R(j)<<2(R(state)), R(t2)
	MOVB R(t2), R(i)<<2(R(state))
	MOVB R(t), R(j)<<2(R(state))

	// dst[k] = src[k] ^ state[state[i] + state[j]]
	ADD R(t2), R(t)
	AND $0xff, R(t)
	MOVBU R(t)<<2(R(state)), R(t)
	MOVBU R(k)<<0(R(src)), R(t2)
	EOR R(t), R(t2)
	MOVB R(t2), R(k)<<0(R(dst))

	ADD $1, R(k)
	CMP R(k), R(n)
	BNE loop

done:
	MOVB R(i), (R(pi))
	MOVB R(j), (R(pj))
	RET
