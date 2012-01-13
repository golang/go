// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package strconv

import "math"

// An extFloat represents an extended floating-point number, with more
// precision than a float64. It does not try to save bits: the
// number represented by the structure is mant*(2^exp), with a negative
// sign if neg is true.
type extFloat struct {
	mant uint64
	exp  int
	neg  bool
}

// Powers of ten taken from double-conversion library.
// http://code.google.com/p/double-conversion/
const (
	firstPowerOfTen = -348
	stepPowerOfTen  = 8
)

var smallPowersOfTen = [...]extFloat{
	{1 << 63, -63, false},        // 1
	{0xa << 60, -60, false},      // 1e1
	{0x64 << 57, -57, false},     // 1e2
	{0x3e8 << 54, -54, false},    // 1e3
	{0x2710 << 50, -50, false},   // 1e4
	{0x186a0 << 47, -47, false},  // 1e5
	{0xf4240 << 44, -44, false},  // 1e6
	{0x989680 << 40, -40, false}, // 1e7
}

var powersOfTen = [...]extFloat{
	{0xfa8fd5a0081c0288, -1220, false}, // 10^-348
	{0xbaaee17fa23ebf76, -1193, false}, // 10^-340
	{0x8b16fb203055ac76, -1166, false}, // 10^-332
	{0xcf42894a5dce35ea, -1140, false}, // 10^-324
	{0x9a6bb0aa55653b2d, -1113, false}, // 10^-316
	{0xe61acf033d1a45df, -1087, false}, // 10^-308
	{0xab70fe17c79ac6ca, -1060, false}, // 10^-300
	{0xff77b1fcbebcdc4f, -1034, false}, // 10^-292
	{0xbe5691ef416bd60c, -1007, false}, // 10^-284
	{0x8dd01fad907ffc3c, -980, false},  // 10^-276
	{0xd3515c2831559a83, -954, false},  // 10^-268
	{0x9d71ac8fada6c9b5, -927, false},  // 10^-260
	{0xea9c227723ee8bcb, -901, false},  // 10^-252
	{0xaecc49914078536d, -874, false},  // 10^-244
	{0x823c12795db6ce57, -847, false},  // 10^-236
	{0xc21094364dfb5637, -821, false},  // 10^-228
	{0x9096ea6f3848984f, -794, false},  // 10^-220
	{0xd77485cb25823ac7, -768, false},  // 10^-212
	{0xa086cfcd97bf97f4, -741, false},  // 10^-204
	{0xef340a98172aace5, -715, false},  // 10^-196
	{0xb23867fb2a35b28e, -688, false},  // 10^-188
	{0x84c8d4dfd2c63f3b, -661, false},  // 10^-180
	{0xc5dd44271ad3cdba, -635, false},  // 10^-172
	{0x936b9fcebb25c996, -608, false},  // 10^-164
	{0xdbac6c247d62a584, -582, false},  // 10^-156
	{0xa3ab66580d5fdaf6, -555, false},  // 10^-148
	{0xf3e2f893dec3f126, -529, false},  // 10^-140
	{0xb5b5ada8aaff80b8, -502, false},  // 10^-132
	{0x87625f056c7c4a8b, -475, false},  // 10^-124
	{0xc9bcff6034c13053, -449, false},  // 10^-116
	{0x964e858c91ba2655, -422, false},  // 10^-108
	{0xdff9772470297ebd, -396, false},  // 10^-100
	{0xa6dfbd9fb8e5b88f, -369, false},  // 10^-92
	{0xf8a95fcf88747d94, -343, false},  // 10^-84
	{0xb94470938fa89bcf, -316, false},  // 10^-76
	{0x8a08f0f8bf0f156b, -289, false},  // 10^-68
	{0xcdb02555653131b6, -263, false},  // 10^-60
	{0x993fe2c6d07b7fac, -236, false},  // 10^-52
	{0xe45c10c42a2b3b06, -210, false},  // 10^-44
	{0xaa242499697392d3, -183, false},  // 10^-36
	{0xfd87b5f28300ca0e, -157, false},  // 10^-28
	{0xbce5086492111aeb, -130, false},  // 10^-20
	{0x8cbccc096f5088cc, -103, false},  // 10^-12
	{0xd1b71758e219652c, -77, false},   // 10^-4
	{0x9c40000000000000, -50, false},   // 10^4
	{0xe8d4a51000000000, -24, false},   // 10^12
	{0xad78ebc5ac620000, 3, false},     // 10^20
	{0x813f3978f8940984, 30, false},    // 10^28
	{0xc097ce7bc90715b3, 56, false},    // 10^36
	{0x8f7e32ce7bea5c70, 83, false},    // 10^44
	{0xd5d238a4abe98068, 109, false},   // 10^52
	{0x9f4f2726179a2245, 136, false},   // 10^60
	{0xed63a231d4c4fb27, 162, false},   // 10^68
	{0xb0de65388cc8ada8, 189, false},   // 10^76
	{0x83c7088e1aab65db, 216, false},   // 10^84
	{0xc45d1df942711d9a, 242, false},   // 10^92
	{0x924d692ca61be758, 269, false},   // 10^100
	{0xda01ee641a708dea, 295, false},   // 10^108
	{0xa26da3999aef774a, 322, false},   // 10^116
	{0xf209787bb47d6b85, 348, false},   // 10^124
	{0xb454e4a179dd1877, 375, false},   // 10^132
	{0x865b86925b9bc5c2, 402, false},   // 10^140
	{0xc83553c5c8965d3d, 428, false},   // 10^148
	{0x952ab45cfa97a0b3, 455, false},   // 10^156
	{0xde469fbd99a05fe3, 481, false},   // 10^164
	{0xa59bc234db398c25, 508, false},   // 10^172
	{0xf6c69a72a3989f5c, 534, false},   // 10^180
	{0xb7dcbf5354e9bece, 561, false},   // 10^188
	{0x88fcf317f22241e2, 588, false},   // 10^196
	{0xcc20ce9bd35c78a5, 614, false},   // 10^204
	{0x98165af37b2153df, 641, false},   // 10^212
	{0xe2a0b5dc971f303a, 667, false},   // 10^220
	{0xa8d9d1535ce3b396, 694, false},   // 10^228
	{0xfb9b7cd9a4a7443c, 720, false},   // 10^236
	{0xbb764c4ca7a44410, 747, false},   // 10^244
	{0x8bab8eefb6409c1a, 774, false},   // 10^252
	{0xd01fef10a657842c, 800, false},   // 10^260
	{0x9b10a4e5e9913129, 827, false},   // 10^268
	{0xe7109bfba19c0c9d, 853, false},   // 10^276
	{0xac2820d9623bf429, 880, false},   // 10^284
	{0x80444b5e7aa7cf85, 907, false},   // 10^292
	{0xbf21e44003acdd2d, 933, false},   // 10^300
	{0x8e679c2f5e44ff8f, 960, false},   // 10^308
	{0xd433179d9c8cb841, 986, false},   // 10^316
	{0x9e19db92b4e31ba9, 1013, false},  // 10^324
	{0xeb96bf6ebadf77d9, 1039, false},  // 10^332
	{0xaf87023b9bf0ee6b, 1066, false},  // 10^340
}

// floatBits returns the bits of the float64 that best approximates
// the extFloat passed as receiver. Overflow is set to true if
// the resulting float64 is ±Inf.
func (f *extFloat) floatBits() (bits uint64, overflow bool) {
	flt := &float64info
	f.Normalize()

	exp := f.exp + 63

	// Exponent too small.
	if exp < flt.bias+1 {
		n := flt.bias + 1 - exp
		f.mant >>= uint(n)
		exp += n
	}

	// Extract 1+flt.mantbits bits.
	mant := f.mant >> (63 - flt.mantbits)
	if f.mant&(1<<(62-flt.mantbits)) != 0 {
		// Round up.
		mant += 1
	}

	// Rounding might have added a bit; shift down.
	if mant == 2<<flt.mantbits {
		mant >>= 1
		exp++
	}

	// Infinities.
	if exp-flt.bias >= 1<<flt.expbits-1 {
		goto overflow
	}

	// Denormalized?
	if mant&(1<<flt.mantbits) == 0 {
		exp = flt.bias
	}
	goto out

overflow:
	// ±Inf
	mant = 0
	exp = 1<<flt.expbits - 1 + flt.bias
	overflow = true

out:
	// Assemble bits.
	bits = mant & (uint64(1)<<flt.mantbits - 1)
	bits |= uint64((exp-flt.bias)&(1<<flt.expbits-1)) << flt.mantbits
	if f.neg {
		bits |= 1 << (flt.mantbits + flt.expbits)
	}
	return
}

// Assign sets f to the value of x.
func (f *extFloat) Assign(x float64) {
	if x < 0 {
		x = -x
		f.neg = true
	}
	x, f.exp = math.Frexp(x)
	f.mant = uint64(x * float64(1<<64))
	f.exp -= 64
}

// AssignComputeBounds sets f to the value of x and returns
// lower, upper such that any number in the closed interval
// [lower, upper] is converted back to x.
func (f *extFloat) AssignComputeBounds(x float64) (lower, upper extFloat) {
	// Special cases.
	bits := math.Float64bits(x)
	flt := &float64info
	neg := bits>>(flt.expbits+flt.mantbits) != 0
	expBiased := int(bits>>flt.mantbits) & (1<<flt.expbits - 1)
	mant := bits & (uint64(1)<<flt.mantbits - 1)

	if expBiased == 0 {
		// denormalized.
		f.mant = mant
		f.exp = 1 + flt.bias - int(flt.mantbits)
	} else {
		f.mant = mant | 1<<flt.mantbits
		f.exp = expBiased + flt.bias - int(flt.mantbits)
	}
	f.neg = neg

	upper = extFloat{mant: 2*f.mant + 1, exp: f.exp - 1, neg: f.neg}
	if mant != 0 || expBiased == 1 {
		lower = extFloat{mant: 2*f.mant - 1, exp: f.exp - 1, neg: f.neg}
	} else {
		lower = extFloat{mant: 4*f.mant - 1, exp: f.exp - 2, neg: f.neg}
	}
	return
}

// Normalize normalizes f so that the highest bit of the mantissa is
// set, and returns the number by which the mantissa was left-shifted.
func (f *extFloat) Normalize() uint {
	if f.mant == 0 {
		return 0
	}
	exp_before := f.exp
	for f.mant < (1 << 55) {
		f.mant <<= 8
		f.exp -= 8
	}
	for f.mant < (1 << 63) {
		f.mant <<= 1
		f.exp -= 1
	}
	return uint(exp_before - f.exp)
}

// Multiply sets f to the product f*g: the result is correctly rounded,
// but not normalized.
func (f *extFloat) Multiply(g extFloat) {
	fhi, flo := f.mant>>32, uint64(uint32(f.mant))
	ghi, glo := g.mant>>32, uint64(uint32(g.mant))

	// Cross products.
	cross1 := fhi * glo
	cross2 := flo * ghi

	// f.mant*g.mant is fhi*ghi << 64 + (cross1+cross2) << 32 + flo*glo
	f.mant = fhi*ghi + (cross1 >> 32) + (cross2 >> 32)
	rem := uint64(uint32(cross1)) + uint64(uint32(cross2)) + ((flo * glo) >> 32)
	// Round up.
	rem += (1 << 31)

	f.mant += (rem >> 32)
	f.exp = f.exp + g.exp + 64
}

var uint64pow10 = [...]uint64{
	1, 1e1, 1e2, 1e3, 1e4, 1e5, 1e6, 1e7, 1e8, 1e9,
	1e10, 1e11, 1e12, 1e13, 1e14, 1e15, 1e16, 1e17, 1e18, 1e19,
}

// AssignDecimal sets f to an approximate value of the decimal d. It
// returns true if the value represented by f is guaranteed to be the
// best approximation of d after being rounded to a float64. 
func (f *extFloat) AssignDecimal(d *decimal) (ok bool) {
	const uint64digits = 19
	const errorscale = 8
	mant10, digits := d.atou64()
	exp10 := d.dp - digits
	errors := 0 // An upper bound for error, computed in errorscale*ulp.

	if digits < d.nd {
		// the decimal number was truncated.
		errors += errorscale / 2
	}

	f.mant = mant10
	f.exp = 0
	f.neg = d.neg

	// Multiply by powers of ten.
	i := (exp10 - firstPowerOfTen) / stepPowerOfTen
	if exp10 < firstPowerOfTen || i >= len(powersOfTen) {
		return false
	}
	adjExp := (exp10 - firstPowerOfTen) % stepPowerOfTen

	// We multiply by exp%step
	if digits+adjExp <= uint64digits {
		// We can multiply the mantissa
		f.mant *= uint64(float64pow10[adjExp])
		f.Normalize()
	} else {
		f.Normalize()
		f.Multiply(smallPowersOfTen[adjExp])
		errors += errorscale / 2
	}

	// We multiply by 10 to the exp - exp%step.
	f.Multiply(powersOfTen[i])
	if errors > 0 {
		errors += 1
	}
	errors += errorscale / 2

	// Normalize
	shift := f.Normalize()
	errors <<= shift

	// Now f is a good approximation of the decimal.
	// Check whether the error is too large: that is, if the mantissa
	// is perturbated by the error, the resulting float64 will change.
	// The 64 bits mantissa is 1 + 52 bits for float64 + 11 extra bits.
	//
	// In many cases the approximation will be good enough.
	const denormalExp = -1023 - 63
	flt := &float64info
	var extrabits uint
	if f.exp <= denormalExp {
		extrabits = uint(63 - flt.mantbits + 1 + uint(denormalExp-f.exp))
	} else {
		extrabits = uint(63 - flt.mantbits)
	}

	halfway := uint64(1) << (extrabits - 1)
	mant_extra := f.mant & (1<<extrabits - 1)

	// Do a signed comparison here! If the error estimate could make
	// the mantissa round differently for the conversion to double,
	// then we can't give a definite answer.
	if int64(halfway)-int64(errors) < int64(mant_extra) &&
		int64(mant_extra) < int64(halfway)+int64(errors) {
		return false
	}
	return true
}

// Frexp10 is an analogue of math.Frexp for decimal powers. It scales
// f by an approximate power of ten 10^-exp, and returns exp10, so
// that f*10^exp10 has the same value as the old f, up to an ulp,
// as well as the index of 10^-exp in the powersOfTen table.
// The arguments expMin and expMax constrain the final value of the
// binary exponent of f.
func (f *extFloat) frexp10(expMin, expMax int) (exp10, index int) {
	// it is illegal to call this function with a too restrictive exponent range.
	if expMax-expMin <= 25 {
		panic("strconv: invalid exponent range")
	}
	// Find power of ten such that x * 10^n has a binary exponent
	// between expMin and expMax
	approxExp10 := -(f.exp + 100) * 28 / 93 // log(10)/log(2) is close to 93/28.
	i := (approxExp10 - firstPowerOfTen) / stepPowerOfTen
Loop:
	for {
		exp := f.exp + powersOfTen[i].exp + 64
		switch {
		case exp < expMin:
			i++
		case exp > expMax:
			i--
		default:
			break Loop
		}
	}
	// Apply the desired decimal shift on f. It will have exponent
	// in the desired range. This is multiplication by 10^-exp10.
	f.Multiply(powersOfTen[i])

	return -(firstPowerOfTen + i*stepPowerOfTen), i
}

// frexp10Many applies a common shift by a power of ten to a, b, c.
func frexp10Many(expMin, expMax int, a, b, c *extFloat) (exp10 int) {
	exp10, i := c.frexp10(expMin, expMax)
	a.Multiply(powersOfTen[i])
	b.Multiply(powersOfTen[i])
	return
}

// ShortestDecimal stores in d the shortest decimal representation of f
// which belongs to the open interval (lower, upper), where f is supposed
// to lie. It returns false whenever the result is unsure. The implementation
// uses the Grisu3 algorithm.
func (f *extFloat) ShortestDecimal(d *decimal, lower, upper *extFloat) bool {
	if f.mant == 0 {
		d.d[0] = '0'
		d.nd = 1
		d.dp = 0
		d.neg = f.neg
	}
	const minExp = -60
	const maxExp = -32
	upper.Normalize()
	// Uniformize exponents.
	if f.exp > upper.exp {
		f.mant <<= uint(f.exp - upper.exp)
		f.exp = upper.exp
	}
	if lower.exp > upper.exp {
		lower.mant <<= uint(lower.exp - upper.exp)
		lower.exp = upper.exp
	}

	exp10 := frexp10Many(minExp, maxExp, lower, f, upper)
	// Take a safety margin due to rounding in frexp10Many, but we lose precision.
	upper.mant++
	lower.mant--

	// The shortest representation of f is either rounded up or down, but
	// in any case, it is a truncation of upper.
	shift := uint(-upper.exp)
	integer := uint32(upper.mant >> shift)
	fraction := upper.mant - (uint64(integer) << shift)

	// How far we can go down from upper until the result is wrong.
	allowance := upper.mant - lower.mant
	// How far we should go to get a very precise result.
	targetDiff := upper.mant - f.mant

	// Count integral digits: there are at most 10.
	var integerDigits int
	for i, pow := range uint64pow10 {
		if uint64(integer) >= pow {
			integerDigits = i + 1
		}
	}
	for i := 0; i < integerDigits; i++ {
		pow := uint64pow10[integerDigits-i-1]
		digit := integer / uint32(pow)
		d.d[i] = byte(digit + '0')
		integer -= digit * uint32(pow)
		// evaluate whether we should stop.
		if currentDiff := uint64(integer)<<shift + fraction; currentDiff < allowance {
			d.nd = i + 1
			d.dp = integerDigits + exp10
			d.neg = f.neg
			// Sometimes allowance is so large the last digit might need to be
			// decremented to get closer to f.
			return adjustLastDigit(d, currentDiff, targetDiff, allowance, pow<<shift, 2)
		}
	}
	d.nd = integerDigits
	d.dp = d.nd + exp10
	d.neg = f.neg

	// Compute digits of the fractional part. At each step fraction does not
	// overflow. The choice of minExp implies that fraction is less than 2^60.
	var digit int
	multiplier := uint64(1)
	for {
		fraction *= 10
		multiplier *= 10
		digit = int(fraction >> shift)
		d.d[d.nd] = byte(digit + '0')
		d.nd++
		fraction -= uint64(digit) << shift
		if fraction < allowance*multiplier {
			// We are in the admissible range. Note that if allowance is about to
			// overflow, that is, allowance > 2^64/10, the condition is automatically
			// true due to the limited range of fraction.
			return adjustLastDigit(d,
				fraction, targetDiff*multiplier, allowance*multiplier,
				1<<shift, multiplier*2)
		}
	}
	return false
}

// adjustLastDigit modifies d = x-currentDiff*ε, to get closest to 
// d = x-targetDiff*ε, without becoming smaller than x-maxDiff*ε.
// It assumes that a decimal digit is worth ulpDecimal*ε, and that
// all data is known with a error estimate of ulpBinary*ε.
func adjustLastDigit(d *decimal, currentDiff, targetDiff, maxDiff, ulpDecimal, ulpBinary uint64) bool {
	if ulpDecimal < 2*ulpBinary {
		// Appromixation is too wide.
		return false
	}
	for currentDiff+ulpDecimal/2+ulpBinary < targetDiff {
		d.d[d.nd-1]--
		currentDiff += ulpDecimal
	}
	if currentDiff+ulpDecimal <= targetDiff+ulpDecimal/2+ulpBinary {
		// we have two choices, and don't know what to do.
		return false
	}
	if currentDiff < ulpBinary || currentDiff > maxDiff-ulpBinary {
		// we went too far
		return false
	}
	if d.nd == 1 && d.d[0] == '0' {
		// the number has actually reached zero.
		d.nd = 0
		d.dp = 0
	}
	return true
}
