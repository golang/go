// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include "../runtime/textflag.h"

TEXT ·Asin(SB),NOSPLIT,$0
	BR ·asin(SB)

TEXT ·Acos(SB),NOSPLIT,$0
	BR ·acos(SB)

TEXT ·Atan2(SB),NOSPLIT,$0
	BR ·atan2(SB)

TEXT ·Atan(SB),NOSPLIT,$0
	BR ·atan(SB)

TEXT ·Exp2(SB),NOSPLIT,$0
	BR ·exp2(SB)

TEXT ·Expm1(SB),NOSPLIT,$0
	BR ·expm1(SB)

TEXT ·Exp(SB),NOSPLIT,$0
	BR ·exp(SB)

TEXT ·Floor(SB),NOSPLIT,$0
	BR ·floor(SB)

TEXT ·Ceil(SB),NOSPLIT,$0
	BR ·ceil(SB)

TEXT ·Trunc(SB),NOSPLIT,$0
	BR ·trunc(SB)

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

TEXT ·Cos(SB),NOSPLIT,$0
	BR ·cos(SB)

TEXT ·Tan(SB),NOSPLIT,$0
	BR ·tan(SB)
