// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cmplx

import (
	"math"
	"math/bits"
)

// The original C code, the long comment, and the constants
// below are from http://netlib.sandia.gov/cephes/c9x-complex/clog.c.
// The go code is a simplified version of the original C.
//
// Cephes Math Library Release 2.8:  June, 2000
// Copyright 1984, 1987, 1989, 1992, 2000 by Stephen L. Moshier
//
// The readme file at http://netlib.sandia.gov/cephes/ says:
//    Some software in this archive may be from the book _Methods and
// Programs for Mathematical Functions_ (Prentice-Hall or Simon & Schuster
// International, 1989) or from the Cephes Mathematical Library, a
// commercial product. In either event, it is copyrighted by the author.
// What you see here may be used freely but it comes with no support or
// guarantee.
//
//   The two known misprints in the book are repaired here in the
// source listings for the gamma function and the incomplete beta
// integral.
//
//   Stephen L. Moshier
//   moshier@na-net.ornl.gov

// Complex circular tangent
//
// DESCRIPTION:
//
// If
//     z = x + iy,
//
// then
//
//           sin 2x  +  i sinh 2y
//     w  =  --------------------.
//            cos 2x  +  cosh 2y
//
// On the real axis the denominator is zero at odd multiples
// of PI/2. The denominator is evaluated by its Taylor
// series near these points.
//
// ctan(z) = -i ctanh(iz).
//
// ACCURACY:
//
//                      Relative error:
// arithmetic   domain     # trials      peak         rms
//    DEC       -10,+10      5200       7.1e-17     1.6e-17
//    IEEE      -10,+10     30000       7.2e-16     1.2e-16
// Also tested by ctan * ccot = 1 and catan(ctan(z))  =  z.

// Tan returns the tangent of x.
func Tan(x complex128) complex128 {
	switch re, im := real(x), imag(x); {
	case math.IsInf(im, 0):
		switch {
		case math.IsInf(re, 0) || math.IsNaN(re):
			return complex(math.Copysign(0, re), math.Copysign(1, im))
		}
		return complex(math.Copysign(0, math.Sin(2*re)), math.Copysign(1, im))
	case re == 0 && math.IsNaN(im):
		return x
	}
	d := math.Cos(2*real(x)) + math.Cosh(2*imag(x))
	if math.Abs(d) < 0.25 {
		d = tanSeries(x)
	}
	if d == 0 {
		return Inf()
	}
	return complex(math.Sin(2*real(x))/d, math.Sinh(2*imag(x))/d)
}

// Complex hyperbolic tangent
//
// DESCRIPTION:
//
// tanh z = (sinh 2x  +  i sin 2y) / (cosh 2x + cos 2y) .
//
// ACCURACY:
//
//                      Relative error:
// arithmetic   domain     # trials      peak         rms
//    IEEE      -10,+10     30000       1.7e-14     2.4e-16

// Tanh returns the hyperbolic tangent of x.
func Tanh(x complex128) complex128 {
	switch re, im := real(x), imag(x); {
	case math.IsInf(re, 0):
		switch {
		case math.IsInf(im, 0) || math.IsNaN(im):
			return complex(math.Copysign(1, re), math.Copysign(0, im))
		}
		return complex(math.Copysign(1, re), math.Copysign(0, math.Sin(2*im)))
	case im == 0 && math.IsNaN(re):
		return x
	}
	d := math.Cosh(2*real(x)) + math.Cos(2*imag(x))
	if d == 0 {
		return Inf()
	}
	return complex(math.Sinh(2*real(x))/d, math.Sin(2*imag(x))/d)
}

// reducePi reduces the input argument x to the range (-Pi/2, Pi/2].
// x must be greater than or equal to 0. For small arguments it
// uses Cody-Waite reduction in 3 float64 parts based on:
// "Elementary Function Evaluation:  Algorithms and Implementation"
// Jean-Michel Muller, 1997.
// For very large arguments it uses Payne-Hanek range reduction based on:
// "ARGUMENT REDUCTION FOR HUGE ARGUMENTS: Good to the Last Bit"
// K. C. Ng et al, March 24, 1992.
func reducePi(x float64) float64 {
	// reduceThreshold is the maximum value of x where the reduction using
	// Cody-Waite reduction still gives accurate results. This threshold
	// is set by t*PIn being representable as a float64 without error
	// where t is given by t = floor(x * (1 / Pi)) and PIn are the leading partial
	// terms of Pi. Since the leading terms, PI1 and PI2 below, have 30 and 32
	// trailing zero bits respectively, t should have less than 30 significant bits.
	//	t < 1<<30  -> floor(x*(1/Pi)+0.5) < 1<<30 -> x < (1<<30-1) * Pi - 0.5
	// So, conservatively we can take x < 1<<30.
	const reduceThreshold float64 = 1 << 30
	if math.Abs(x) < reduceThreshold {
		// Use Cody-Waite reduction in three parts.
		const (
			// PI1, PI2 and PI3 comprise an extended precision value of PI
			// such that PI ~= PI1 + PI2 + PI3. The parts are chosen so
			// that PI1 and PI2 have an approximately equal number of trailing
			// zero bits. This ensures that t*PI1 and t*PI2 are exact for
			// large integer values of t. The full precision PI3 ensures the
			// approximation of PI is accurate to 102 bits to handle cancellation
			// during subtraction.
			PI1 = 3.141592502593994      // 0x400921fb40000000
			PI2 = 1.5099578831723193e-07 // 0x3e84442d00000000
			PI3 = 1.0780605716316238e-14 // 0x3d08469898cc5170
		)
		t := x / math.Pi
		t += 0.5
		t = float64(int64(t)) // int64(t) = the multiple
		return ((x - t*PI1) - t*PI2) - t*PI3
	}
	// Must apply Payne-Hanek range reduction
	const (
		mask     = 0x7FF
		shift    = 64 - 11 - 1
		bias     = 1023
		fracMask = 1<<shift - 1
	)
	// Extract out the integer and exponent such that,
	// x = ix * 2 ** exp.
	ix := math.Float64bits(x)
	exp := int(ix>>shift&mask) - bias - shift
	ix &= fracMask
	ix |= 1 << shift

	// mPi is the binary digits of 1/Pi as a uint64 array,
	// that is, 1/Pi = Sum mPi[i]*2^(-64*i).
	// 19 64-bit digits give 1216 bits of precision
	// to handle the largest possible float64 exponent.
	var mPi = [...]uint64{
		0x0000000000000000,
		0x517cc1b727220a94,
		0xfe13abe8fa9a6ee0,
		0x6db14acc9e21c820,
		0xff28b1d5ef5de2b0,
		0xdb92371d2126e970,
		0x0324977504e8c90e,
		0x7f0ef58e5894d39f,
		0x74411afa975da242,
		0x74ce38135a2fbf20,
		0x9cc8eb1cc1a99cfa,
		0x4e422fc5defc941d,
		0x8ffc4bffef02cc07,
		0xf79788c5ad05368f,
		0xb69b3f6793e584db,
		0xa7a31fb34f2ff516,
		0xba93dd63f5f2f8bd,
		0x9e839cfbc5294975,
		0x35fdafd88fc6ae84,
		0x2b0198237e3db5d5,
	}
	// Use the exponent to extract the 3 appropriate uint64 digits from mPi,
	// B ~ (z0, z1, z2), such that the product leading digit has the exponent -64.
	// Note, exp >= 50 since x >= reduceThreshold and exp < 971 for maximum float64.
	digit, bitshift := uint(exp+64)/64, uint(exp+64)%64
	z0 := (mPi[digit] << bitshift) | (mPi[digit+1] >> (64 - bitshift))
	z1 := (mPi[digit+1] << bitshift) | (mPi[digit+2] >> (64 - bitshift))
	z2 := (mPi[digit+2] << bitshift) | (mPi[digit+3] >> (64 - bitshift))
	// Multiply mantissa by the digits and extract the upper two digits (hi, lo).
	z2hi, _ := bits.Mul64(z2, ix)
	z1hi, z1lo := bits.Mul64(z1, ix)
	z0lo := z0 * ix
	lo, c := bits.Add64(z1lo, z2hi, 0)
	hi, _ := bits.Add64(z0lo, z1hi, c)
	// Find the magnitude of the fraction.
	lz := uint(bits.LeadingZeros64(hi))
	e := uint64(bias - (lz + 1))
	// Clear implicit mantissa bit and shift into place.
	hi = (hi << (lz + 1)) | (lo >> (64 - (lz + 1)))
	hi >>= 64 - shift
	// Include the exponent and convert to a float.
	hi |= e << shift
	x = math.Float64frombits(hi)
	// map to (-Pi/2, Pi/2]
	if x > 0.5 {
		x--
	}
	return math.Pi * x
}

// Taylor series expansion for cosh(2y) - cos(2x)
func tanSeries(z complex128) float64 {
	const MACHEP = 1.0 / (1 << 53)
	x := math.Abs(2 * real(z))
	y := math.Abs(2 * imag(z))
	x = reducePi(x)
	x = x * x
	y = y * y
	x2 := 1.0
	y2 := 1.0
	f := 1.0
	rn := 0.0
	d := 0.0
	for {
		rn++
		f *= rn
		rn++
		f *= rn
		x2 *= x
		y2 *= y
		t := y2 + x2
		t /= f
		d += t

		rn++
		f *= rn
		rn++
		f *= rn
		x2 *= x
		y2 *= y
		t = y2 - x2
		t /= f
		d += t
		if !(math.Abs(t/d) > MACHEP) {
			// Caution: Use ! and > instead of <= for correct behavior if t/d is NaN.
			// See issue 17577.
			break
		}
	}
	return d
}

// Complex circular cotangent
//
// DESCRIPTION:
//
// If
//     z = x + iy,
//
// then
//
//           sin 2x  -  i sinh 2y
//     w  =  --------------------.
//            cosh 2y  -  cos 2x
//
// On the real axis, the denominator has zeros at even
// multiples of PI/2.  Near these points it is evaluated
// by a Taylor series.
//
// ACCURACY:
//
//                      Relative error:
// arithmetic   domain     # trials      peak         rms
//    DEC       -10,+10      3000       6.5e-17     1.6e-17
//    IEEE      -10,+10     30000       9.2e-16     1.2e-16
// Also tested by ctan * ccot = 1 + i0.

// Cot returns the cotangent of x.
func Cot(x complex128) complex128 {
	d := math.Cosh(2*imag(x)) - math.Cos(2*real(x))
	if math.Abs(d) < 0.25 {
		d = tanSeries(x)
	}
	if d == 0 {
		return Inf()
	}
	return complex(math.Sin(2*real(x))/d, -math.Sinh(2*imag(x))/d)
}
