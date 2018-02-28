// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips mipsle

#include "textflag.h"

TEXT ·SwapInt32(SB),NOSPLIT,$0-12
	JMP	runtime∕internal∕atomic·Xchg(SB)

TEXT ·SwapUint32(SB),NOSPLIT,$0-12
	JMP	runtime∕internal∕atomic·Xchg(SB)

TEXT ·SwapInt64(SB),NOSPLIT,$0-20
	JMP	runtime∕internal∕atomic·Xchg64(SB)

TEXT ·SwapUint64(SB),NOSPLIT,$0-20
	JMP	runtime∕internal∕atomic·Xchg64(SB)

TEXT ·SwapUintptr(SB),NOSPLIT,$0-12
	JMP	runtime∕internal∕atomic·Xchg(SB)

TEXT ·CompareAndSwapInt32(SB),NOSPLIT,$0-13
	JMP	runtime∕internal∕atomic·Cas(SB)

TEXT ·CompareAndSwapUint32(SB),NOSPLIT,$0-13
	JMP	runtime∕internal∕atomic·Cas(SB)

TEXT ·CompareAndSwapUintptr(SB),NOSPLIT,$0-13
	JMP	runtime∕internal∕atomic·Cas(SB)

TEXT ·CompareAndSwapInt64(SB),NOSPLIT,$0-21
	JMP	runtime∕internal∕atomic·Cas64(SB)

TEXT ·CompareAndSwapUint64(SB),NOSPLIT,$0-21
	JMP	runtime∕internal∕atomic·Cas64(SB)

TEXT ·AddInt32(SB),NOSPLIT,$0-12
	JMP	runtime∕internal∕atomic·Xadd(SB)

TEXT ·AddUint32(SB),NOSPLIT,$0-12
	JMP	runtime∕internal∕atomic·Xadd(SB)

TEXT ·AddUintptr(SB),NOSPLIT,$0-12
	JMP	runtime∕internal∕atomic·Xadd(SB)

TEXT ·AddInt64(SB),NOSPLIT,$0-20
	JMP	runtime∕internal∕atomic·Xadd64(SB)

TEXT ·AddUint64(SB),NOSPLIT,$0-20
	JMP	runtime∕internal∕atomic·Xadd64(SB)

TEXT ·LoadInt32(SB),NOSPLIT,$0-8
	JMP	runtime∕internal∕atomic·Load(SB)

TEXT ·LoadUint32(SB),NOSPLIT,$0-8
	JMP	runtime∕internal∕atomic·Load(SB)

TEXT ·LoadInt64(SB),NOSPLIT,$0-12
	JMP	runtime∕internal∕atomic·Load64(SB)

TEXT ·LoadUint64(SB),NOSPLIT,$0-12
	JMP	runtime∕internal∕atomic·Load64(SB)

TEXT ·LoadUintptr(SB),NOSPLIT,$0-8
	JMP	runtime∕internal∕atomic·Load(SB)

TEXT ·LoadPointer(SB),NOSPLIT,$0-8
	JMP	runtime∕internal∕atomic·Load(SB)

TEXT ·StoreInt32(SB),NOSPLIT,$0-8
	JMP	runtime∕internal∕atomic·Store(SB)

TEXT ·StoreUint32(SB),NOSPLIT,$0-8
	JMP	runtime∕internal∕atomic·Store(SB)

TEXT ·StoreInt64(SB),NOSPLIT,$0-12
	JMP	runtime∕internal∕atomic·Store64(SB)

TEXT ·StoreUint64(SB),NOSPLIT,$0-12
	JMP	runtime∕internal∕atomic·Store64(SB)

TEXT ·StoreUintptr(SB),NOSPLIT,$0-8
	JMP	runtime∕internal∕atomic·Store(SB)
