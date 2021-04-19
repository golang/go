// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build ppc64 ppc64le

#include "textflag.h"

TEXT ·Asin(SB),NOSPLIT,$0
	BR ·asin(SB)

TEXT ·Acos(SB),NOSPLIT,$0
	BR ·acos(SB)

TEXT ·Atan2(SB),NOSPLIT,$0
	BR ·atan2(SB)

TEXT ·Atan(SB),NOSPLIT,$0
	BR ·atan(SB)

TEXT ·Dim(SB),NOSPLIT,$0
	BR ·dim(SB)

TEXT ·Min(SB),NOSPLIT,$0
	BR ·min(SB)

TEXT ·Max(SB),NOSPLIT,$0
	BR ·max(SB)

TEXT ·Exp2(SB),NOSPLIT,$0
	BR ·exp2(SB)

TEXT ·Expm1(SB),NOSPLIT,$0
	BR ·expm1(SB)

TEXT ·Exp(SB),NOSPLIT,$0
	BR ·exp(SB)

TEXT ·Frexp(SB),NOSPLIT,$0
	BR ·frexp(SB)

TEXT ·Hypot(SB),NOSPLIT,$0
	BR ·hypot(SB)

TEXT ·Ldexp(SB),NOSPLIT,$0
	BR ·ldexp(SB)

TEXT ·Log10(SB),NOSPLIT,$0
	BR ·log10(SB)

TEXT ·Log2(SB),NOSPLIT,$0
	BR ·log2(SB)

TEXT ·Log1p(SB),NOSPLIT,$0
	BR ·log1p(SB)

TEXT ·Log(SB),NOSPLIT,$0
	BR ·log(SB)

TEXT ·Modf(SB),NOSPLIT,$0
	BR ·modf(SB)

TEXT ·Mod(SB),NOSPLIT,$0
	BR ·mod(SB)

TEXT ·Remainder(SB),NOSPLIT,$0
	BR ·remainder(SB)

TEXT ·Sincos(SB),NOSPLIT,$0
	BR ·sincos(SB)

TEXT ·Sin(SB),NOSPLIT,$0
	BR ·sin(SB)

TEXT ·Sinh(SB),NOSPLIT,$0
	BR ·sinh(SB)

TEXT ·Cos(SB),NOSPLIT,$0
	BR ·cos(SB)

TEXT ·Cosh(SB),NOSPLIT,$0
	BR ·cosh(SB)

TEXT ·Tan(SB),NOSPLIT,$0
	BR ·tan(SB)

TEXT ·Tanh(SB),NOSPLIT,$0
	BR ·tanh(SB)
