// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build arm64

#include "textflag.h"

TEXT ·Asin(SB),NOSPLIT,$0
	B ·asin(SB)

TEXT ·Acos(SB),NOSPLIT,$0
	B ·acos(SB)

TEXT ·Atan2(SB),NOSPLIT,$0
	B ·atan2(SB)

TEXT ·Atan(SB),NOSPLIT,$0
	B ·atan(SB)

TEXT ·Dim(SB),NOSPLIT,$0
	B ·dim(SB)

TEXT ·Min(SB),NOSPLIT,$0
	B ·min(SB)

TEXT ·Max(SB),NOSPLIT,$0
	B ·max(SB)

TEXT ·Exp2(SB),NOSPLIT,$0
	B ·exp2(SB)

TEXT ·Expm1(SB),NOSPLIT,$0
	B ·expm1(SB)

TEXT ·Exp(SB),NOSPLIT,$0
	B ·exp(SB)

TEXT ·Floor(SB),NOSPLIT,$0
	B ·floor(SB)

TEXT ·Ceil(SB),NOSPLIT,$0
	B ·ceil(SB)

TEXT ·Trunc(SB),NOSPLIT,$0
	B ·trunc(SB)

TEXT ·Frexp(SB),NOSPLIT,$0
	B ·frexp(SB)

TEXT ·Hypot(SB),NOSPLIT,$0
	B ·hypot(SB)

TEXT ·Ldexp(SB),NOSPLIT,$0
	B ·ldexp(SB)

TEXT ·Log10(SB),NOSPLIT,$0
	B ·log10(SB)

TEXT ·Log2(SB),NOSPLIT,$0
	B ·log2(SB)

TEXT ·Log1p(SB),NOSPLIT,$0
	B ·log1p(SB)

TEXT ·Log(SB),NOSPLIT,$0
	B ·log(SB)

TEXT ·Modf(SB),NOSPLIT,$0
	B ·modf(SB)

TEXT ·Mod(SB),NOSPLIT,$0
	B ·mod(SB)

TEXT ·Remainder(SB),NOSPLIT,$0
	B ·remainder(SB)

TEXT ·Sincos(SB),NOSPLIT,$0
	B ·sincos(SB)

TEXT ·Sin(SB),NOSPLIT,$0
	B ·sin(SB)

TEXT ·Cos(SB),NOSPLIT,$0
	B ·cos(SB)

TEXT ·Tan(SB),NOSPLIT,$0
	B ·tan(SB)
