// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv

// Binary to decimal conversion using the Dragonbox algorithm by Junekey Jeon.
//
// Fixed precision format is not supported by the Dragonbox algorithm
// so we continue to use Ryū-printf for this purpose.
// See https://github.com/jk-jeon/dragonbox/issues/38 for more details.
//
// For binary to decimal rounding, uses round to nearest, tie to even.
// For decimal to binary rounding, assumes round to nearest, tie to even.
//
// The original paper by Junekey Jeon can be found at:
// https://github.com/jk-jeon/dragonbox/blob/d5dc40ae6a3f1a4559cda816738df2d6255b4e24/other_files/Dragonbox.pdf
//
// The reference implementation in C++ by Junekey Jeon can be found at:
// https://github.com/jk-jeon/dragonbox/blob/6c7c925b571d54486b9ffae8d9d18a822801cbda/subproject/simple/include/simple_dragonbox.h

// dragonboxFtoa computes the decimal significand and exponent
// from the binary significand and exponent using the Dragonbox algorithm
// and formats the decimal floating point number in d.
func dboxFtoa(d *decimalSlice, mant uint64, exp int, denorm bool, bitSize int) {
	if bitSize == 32 {
		dboxFtoa32(d, uint32(mant), exp, denorm)
		return
	}
	dboxFtoa64(d, mant, exp, denorm)
}

func dboxFtoa64(d *decimalSlice, mant uint64, exp int, denorm bool) {
	if mant == 1<<float64MantBits && !denorm {
		// Algorithm 5.6 (page 24).
		k0 := -mulLog10_2MinusLog10_4Over3(exp)
		φ, β := dboxPow64(k0, exp)
		xi, zi := dboxRange64(φ, β)
		if exp != 2 && exp != 3 {
			xi++
		}
		q := zi / 10
		if xi <= q*10 {
			q, zeros := trimZeros(q)
			dboxDigits(d, q, -k0+1+zeros)
			return
		}
		yru := dboxRoundUp64(φ, β)
		if exp == -77 && yru%2 != 0 {
			yru--
		} else if yru < xi {
			yru++
		}
		dboxDigits(d, yru, -k0)
		return
	}

	// κ = 2 for float64 (section 5.1.3)
	const (
		κ     = 2
		p10κ  = 100       // 10**κ
		p10κ1 = p10κ * 10 // 10**(κ+1)
	)

	// Algorithm 5.2 (page 15).
	k0 := -mulLog10_2(exp)
	φ, β := dboxPow64(κ+k0, exp)
	zi, exact := dboxMulPow64(uint64(mant*2+1)<<β, φ)
	s, r := zi/p10κ1, uint32(zi%p10κ1)
	δi := dboxDelta64(φ, β)

	if r < δi {
		if r != 0 || !exact || mant%2 == 0 {
			s, zeros := trimZeros(s)
			dboxDigits(d, s, -k0+1+zeros)
			return
		}
		s--
		r = p10κ * 10
	} else if r == δi {
		parity, exact := dboxParity64(uint64(mant*2-1), φ, β)
		if parity || (exact && mant%2 == 0) {
			s, zeros := trimZeros(s)
			dboxDigits(d, s, -k0+1+zeros)
			return
		}
	}

	// Algorithm 5.4 (page 18).
	D := r + p10κ/2 - δi/2
	t, ρ := D/p10κ, D%p10κ
	yru := 10*s + uint64(t)
	if ρ == 0 {
		parity, exact := dboxParity64(mant*2, φ, β)
		if parity != ((D-p10κ/2)%2 != 0) || exact && yru%2 != 0 {
			yru--
		}
	}
	dboxDigits(d, yru, -k0)
}

// Almost identical to dragonboxFtoa64.
// This is kept as a separate copy to minimize runtime overhead.
func dboxFtoa32(d *decimalSlice, mant uint32, exp int, denorm bool) {
	if mant == 1<<float32MantBits && !denorm {
		// Algorithm 5.6 (page 24).
		k0 := -mulLog10_2MinusLog10_4Over3(exp)
		φ, β := dboxPow32(k0, exp)
		xi, zi := dboxRange32(φ, β)
		if exp != 2 && exp != 3 {
			xi++
		}
		q := zi / 10
		if xi <= q*10 {
			q, zeros := trimZeros(uint64(q))
			dboxDigits(d, q, -k0+1+zeros)
			return
		}
		yru := dboxRoundUp32(φ, β)
		if exp == -77 && yru%2 != 0 {
			yru--
		} else if yru < xi {
			yru++
		}
		dboxDigits(d, uint64(yru), -k0)
		return
	}

	// κ = 1 for float32 (section 5.1.3)
	const (
		κ     = 1
		p10κ  = 10
		p10κ1 = p10κ * 10
	)

	// Algorithm 5.2 (page 15).
	k0 := -mulLog10_2(exp)
	φ, β := dboxPow32(κ+k0, exp)
	zi, exact := dboxMulPow32(uint32(mant*2+1)<<β, φ)
	s, r := zi/p10κ1, uint32(zi%p10κ1)
	δi := dboxDelta32(φ, β)

	if r < δi {
		if r != 0 || !exact || mant%2 == 0 {
			s, zeros := trimZeros(uint64(s))
			dboxDigits(d, s, -k0+1+zeros)
			return
		}
		s--
		r = p10κ * 10
	} else if r == δi {
		parity, exact := dboxParity32(uint32(mant*2-1), φ, β)
		if parity || (exact && mant%2 == 0) {
			s, zeros := trimZeros(uint64(s))
			dboxDigits(d, s, -k0+1+zeros)
			return
		}
	}

	// Algorithm 5.4 (page 18).
	D := r + p10κ/2 - δi/2
	t, ρ := D/p10κ, D%p10κ
	yru := 10*s + uint32(t)
	if ρ == 0 {
		parity, exact := dboxParity32(mant*2, φ, β)
		if parity != ((D-p10κ/2)%2 != 0) || exact && yru%2 != 0 {
			yru--
		}
	}
	dboxDigits(d, uint64(yru), -k0)
}

// dboxDigits emits decimal digits of mant in d for float64
// and adjusts the decimal point based on exp.
func dboxDigits(d *decimalSlice, mant uint64, exp int) {
	i := formatBase10(d.d, mant)
	d.d = d.d[i:]
	d.nd = len(d.d)
	d.dp = d.nd + exp
}

// uadd128 returns the full 128 bits of u + n.
func uadd128(u uint128, n uint64) uint128 {
	sum := uint64(u.Lo + n)
	// Check if lo is wrapped around.
	if sum < u.Lo {
		u.Hi++
	}
	u.Lo = sum
	return u
}

// umul64 returns the full 64 bits of x * y.
func umul64(x, y uint32) uint64 {
	return uint64(x) * uint64(y)
}

// umul96Upper64 returns the upper 64 bits (out of 96 bits) of x * y.
func umul96Upper64(x uint32, y uint64) uint64 {
	yh := uint32(y >> 32)
	yl := uint32(y)

	xyh := umul64(x, yh)
	xyl := umul64(x, yl)

	return xyh + (xyl >> 32)
}

// umul96Lower64 returns the lower 64 bits (out of 96 bits) of x * y.
func umul96Lower64(x uint32, y uint64) uint64 {
	return uint64(uint64(x) * y)
}

// umul128Upper64 returns the upper 64 bits (out of 128 bits) of x * y.
func umul128Upper64(x, y uint64) uint64 {
	a := uint32(x >> 32)
	b := uint32(x)
	c := uint32(y >> 32)
	d := uint32(y)

	ac := umul64(a, c)
	bc := umul64(b, c)
	ad := umul64(a, d)
	bd := umul64(b, d)

	intermediate := (bd >> 32) + uint64(uint32(ad)) + uint64(uint32(bc))

	return ac + (intermediate >> 32) + (ad >> 32) + (bc >> 32)
}

// umul192Upper128 returns the upper 128 bits (out of 192 bits) of x * y.
func umul192Upper128(x uint64, y uint128) uint128 {
	r := umul128(x, y.Hi)
	t := umul128Upper64(x, y.Lo)
	return uadd128(r, t)
}

// umul192Lower128 returns the lower 128 bits (out of 192 bits) of x * y.
func umul192Lower128(x uint64, y uint128) uint128 {
	high := x * y.Hi
	highLow := umul128(x, y.Lo)
	return uint128{uint64(high + highLow.Hi), highLow.Lo}
}

// dboxMulPow64 computes x^(i), y^(i), z^(i)
// from the precomputed value of φ̃k for float64
// and also checks if x^(f), y^(f), z^(f) == 0 (section 5.2.1).
func dboxMulPow64(u uint64, phi uint128) (intPart uint64, isInt bool) {
	r := umul192Upper128(u, phi)
	intPart = r.Hi
	isInt = r.Lo == 0
	return
}

// dboxMulPow32 computes x^(i), y^(i), z^(i)
// from the precomputed value of φ̃k for float32
// and also checks if x^(f), y^(f), z^(f) == 0 (section 5.2.1).
func dboxMulPow32(u uint32, phi uint64) (intPart uint32, isInt bool) {
	r := umul96Upper64(u, phi)
	intPart = uint32(r >> 32)
	isInt = uint32(r) == 0
	return
}

// dboxParity64 computes only the parity of x^(i), y^(i), z^(i)
// from the precomputed value of φ̃k for float64
// and also checks if x^(f), y^(f), z^(f) = 0 (section 5.2.1).
func dboxParity64(mant2 uint64, phi uint128, beta int) (parity bool, isInt bool) {
	r := umul192Lower128(mant2, phi)
	parity = ((r.Hi >> (64 - beta)) & 1) != 0
	isInt = ((uint64(r.Hi << beta)) | (r.Lo >> (64 - beta))) == 0
	return
}

// dboxParity32 computes only the parity of x^(i), y^(i), z^(i)
// from the precomputed value of φ̃k for float32
// and also checks if x^(f), y^(f), z^(f) = 0 (section 5.2.1).
func dboxParity32(mant2 uint32, phi uint64, beta int) (parity bool, isInt bool) {
	r := umul96Lower64(mant2, phi)
	parity = ((r >> (64 - beta)) & 1) != 0
	isInt = uint32(r>>(32-beta)) == 0
	return
}

// dboxDelta64 returns δ^(i) from the precomputed value of φ̃k for float64.
func dboxDelta64(φ uint128, β int) uint32 {
	return uint32(φ.Hi >> (64 - 1 - β))
}

// dboxDelta32 returns δ^(i) from the precomputed value of φ̃k for float32.
func dboxDelta32(φ uint64, β int) uint32 {
	return uint32(φ >> (64 - 1 - β))
}

// mulLog10_2MinusLog10_4Over3 computes
// ⌊e*log10(2)-log10(4/3)⌋ = ⌊log10(2^e)-log10(4/3)⌋ (section 6.3).
func mulLog10_2MinusLog10_4Over3(e int) int {
	// e should be in the range [-2985, 2936].
	return (e*631305 - 261663) >> 21
}

const (
	floatMantBits64 = 52 // p = 52 for float64.
	floatMantBits32 = 23 // p = 23 for float32.
)

// dboxRange64 returns the left and right float64 endpoints.
func dboxRange64(φ uint128, β int) (left, right uint64) {
	left = (φ.Hi - (φ.Hi >> (float64MantBits + 2))) >> (64 - float64MantBits - 1 - β)
	right = (φ.Hi + (φ.Hi >> (float64MantBits + 1))) >> (64 - float64MantBits - 1 - β)
	return left, right
}

// dboxRange32 returns the left and right float32 endpoints.
func dboxRange32(φ uint64, β int) (left, right uint32) {
	left = uint32((φ - (φ >> (floatMantBits32 + 2))) >> (64 - floatMantBits32 - 1 - β))
	right = uint32((φ + (φ >> (floatMantBits32 + 1))) >> (64 - floatMantBits32 - 1 - β))
	return left, right
}

// dboxRoundUp64 computes the round up of y (i.e., y^(ru)).
func dboxRoundUp64(phi uint128, beta int) uint64 {
	return (phi.Hi>>(128/2-floatMantBits64-2-beta) + 1) / 2
}

// dboxRoundUp32 computes the round up of y (i.e., y^(ru)).
func dboxRoundUp32(phi uint64, beta int) uint32 {
	return uint32(phi>>(64-floatMantBits32-2-beta)+1) / 2
}

// dboxPow64 gets the precomputed value of φ̃̃k for float64.
func dboxPow64(k, e int) (φ uint128, β int) {
	φ, e1, _ := pow10(k)
	if k < 0 || k > 55 {
		φ.Lo++
	}
	β = e + e1 - 1
	return φ, β
}

// dboxPow32 gets the precomputed value of φ̃̃k for float32.
func dboxPow32(k, e int) (mant uint64, exp int) {
	m, e1, _ := pow10(k)
	if k < 0 || k > 27 {
		m.Hi++
	}
	exp = e + e1 - 1
	return m.Hi, exp
}
