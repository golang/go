// Copyright 2014 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build mips64 mips64le

#include "textflag.h"

TEXT ·Asin(SB), NOSPLIT, $0
	JMP ·asin(SB)

TEXT ·Acos(SB), NOSPLIT, $0
	JMP ·acos(SB)

TEXT ·Asinh(SB), NOSPLIT, $0
	JMP ·asinh(SB)

TEXT ·Acosh(SB), NOSPLIT, $0
	JMP ·acosh(SB)

TEXT ·Atan2(SB), NOSPLIT, $0
	JMP ·atan2(SB)

TEXT ·Atan(SB), NOSPLIT, $0
	JMP ·atan(SB)

TEXT ·Atanh(SB), NOSPLIT, $0
	JMP ·atanh(SB)

TEXT ·Min(SB), NOSPLIT, $0
	JMP ·min(SB)

TEXT ·Max(SB), NOSPLIT, $0
	JMP ·max(SB)

TEXT ·Erf(SB), NOSPLIT, $0
	JMP ·erf(SB)

TEXT ·Erfc(SB), NOSPLIT, $0
	JMP ·erfc(SB)

TEXT ·Exp2(SB), NOSPLIT, $0
	JMP ·exp2(SB)

TEXT ·Expm1(SB), NOSPLIT, $0
	JMP ·expm1(SB)

TEXT ·Exp(SB), NOSPLIT, $0
	JMP ·exp(SB)

TEXT ·Floor(SB), NOSPLIT, $0
	JMP ·floor(SB)

TEXT ·Ceil(SB), NOSPLIT, $0
	JMP ·ceil(SB)

TEXT ·Trunc(SB), NOSPLIT, $0
	JMP ·trunc(SB)

TEXT ·Frexp(SB), NOSPLIT, $0
	JMP ·frexp(SB)

TEXT ·Hypot(SB), NOSPLIT, $0
	JMP ·hypot(SB)

TEXT ·Ldexp(SB), NOSPLIT, $0
	JMP ·ldexp(SB)

TEXT ·Log10(SB), NOSPLIT, $0
	JMP ·log10(SB)

TEXT ·Log2(SB), NOSPLIT, $0
	JMP ·log2(SB)

TEXT ·Log1p(SB), NOSPLIT, $0
	JMP ·log1p(SB)

TEXT ·Log(SB), NOSPLIT, $0
	JMP ·log(SB)

TEXT ·Modf(SB), NOSPLIT, $0
	JMP ·modf(SB)

TEXT ·Mod(SB), NOSPLIT, $0
	JMP ·mod(SB)

TEXT ·Remainder(SB), NOSPLIT, $0
	JMP ·remainder(SB)

TEXT ·Sin(SB), NOSPLIT, $0
	JMP ·sin(SB)

TEXT ·Sinh(SB), NOSPLIT, $0
	JMP ·sinh(SB)

TEXT ·Cos(SB), NOSPLIT, $0
	JMP ·cos(SB)

TEXT ·Cosh(SB), NOSPLIT, $0
	JMP ·cosh(SB)

TEXT ·Sqrt(SB), NOSPLIT, $0
	JMP ·sqrt(SB)

TEXT ·Tan(SB), NOSPLIT, $0
	JMP ·tan(SB)

TEXT ·Tanh(SB), NOSPLIT, $0
	JMP ·tanh(SB)

TEXT ·Cbrt(SB), NOSPLIT, $0
	JMP ·cbrt(SB)

TEXT ·Pow(SB), NOSPLIT, $0
	JMP ·pow(SB)
