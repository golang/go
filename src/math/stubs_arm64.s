// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build arm64

#include "textflag.h"

TEXT ·Asin(SB),NOSPLIT,$0
	B ·asin(SB)

TEXT ·Acos(SB),NOSPLIT,$0
	B ·acos(SB)

TEXT ·Asinh(SB),NOSPLIT,$0
        B ·asinh(SB)

TEXT ·Acosh(SB),NOSPLIT,$0
        B ·acosh(SB)

TEXT ·Atan2(SB),NOSPLIT,$0
	B ·atan2(SB)

TEXT ·Atan(SB),NOSPLIT,$0
	B ·atan(SB)

TEXT ·Atanh(SB),NOSPLIT,$0
	B ·atanh(SB)

TEXT ·Exp2(SB),NOSPLIT,$0
	B ·exp2(SB)

TEXT ·Erf(SB),NOSPLIT,$0
	B ·erf(SB)

TEXT ·Erfc(SB),NOSPLIT,$0
	B ·erfc(SB)

TEXT ·Cbrt(SB),NOSPLIT,$0
	B ·cbrt(SB)

TEXT ·Cosh(SB),NOSPLIT,$0
	B ·cosh(SB)

TEXT ·Expm1(SB),NOSPLIT,$0
	B ·expm1(SB)

TEXT ·Exp(SB),NOSPLIT,$0
	B ·exp(SB)

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

TEXT ·Mod(SB),NOSPLIT,$0
	B ·mod(SB)

TEXT ·Remainder(SB),NOSPLIT,$0
	B ·remainder(SB)

TEXT ·Sin(SB),NOSPLIT,$0
	B ·sin(SB)

TEXT ·Sinh(SB),NOSPLIT,$0
	B ·sinh(SB)

TEXT ·Cos(SB),NOSPLIT,$0
	B ·cos(SB)

TEXT ·Tan(SB),NOSPLIT,$0
	B ·tan(SB)

TEXT ·Tanh(SB),NOSPLIT,$0
	B ·tanh(SB)

TEXT ·Pow(SB),NOSPLIT,$0
	B ·pow(SB)
