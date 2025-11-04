// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mldsa

import (
	"crypto/internal/constanttime"
	"crypto/internal/fips140/sha3"
	"errors"
	"math/bits"
)

const (
	q        = 8380417    // 2²³ - 2¹³ + 1
	R        = 4294967296 // 2³²
	RR       = 2365951    // R² mod q, aka R in the Montgomery domain
	qNegInv  = 4236238847 // -q⁻¹ mod R (q * qNegInv ≡ -1 mod R)
	one      = 4193792    // R mod q, aka 1 in the Montgomery domain
	minusOne = 4186625    // (q - 1) * R mod q, aka -1 in the Montgomery domain
)

// fieldElement is an element n of ℤ_q in the Montgomery domain, represented as
// an integer x in [0, q) such that x ≡ n * R (mod q) where R = 2³².
type fieldElement uint32

var errUnreducedFieldElement = errors.New("mldsa: unreduced field element")

// fieldToMontgomery checks that a value a is < q, and converts it to
// Montgomery form.
func fieldToMontgomery(a uint32) (fieldElement, error) {
	if a >= q {
		return 0, errUnreducedFieldElement
	}
	// a * R² * R⁻¹ ≡ a * R (mod q)
	return fieldMontgomeryMul(fieldElement(a), RR), nil
}

// fieldSubToMontgomery converts a difference a - b to Montgomery form.
// a and b must be < q. (This bound can probably be relaxed.)
func fieldSubToMontgomery(a, b uint32) fieldElement {
	x := a - b + q
	return fieldMontgomeryMul(fieldElement(x), RR)
}

// fieldFromMontgomery converts a value a in Montgomery form back to
// standard representation.
func fieldFromMontgomery(a fieldElement) uint32 {
	// (a * R) * 1 * R⁻¹ ≡ a (mod q)
	return uint32(fieldMontgomeryReduce(uint64(a)))
}

// fieldCenteredMod returns r mod± q, the value r reduced to the range
// [−(q−1)/2, (q−1)/2].
func fieldCenteredMod(r fieldElement) int32 {
	x := int32(fieldFromMontgomery(r))
	// x <= q / 2 ? x : x - q
	return constantTimeSelectLessOrEqual(x, q/2, x, x-q)
}

// fieldInfinityNorm returns the infinity norm ||r||∞ of r, or the absolute
// value of r centered around 0.
func fieldInfinityNorm(r fieldElement) uint32 {
	x := int32(fieldFromMontgomery(r))
	// x <= q / 2 ? x : |x - q|
	// |x - q| = -(x - q) = q - x because x < q => x - q < 0
	return uint32(constantTimeSelectLessOrEqual(x, q/2, x, q-x))
}

// fieldReduceOnce reduces a value a < 2q.
func fieldReduceOnce(a uint32) fieldElement {
	x, b := bits.Sub64(uint64(a), uint64(q), 0)
	return fieldElement(x + b*q)
}

// fieldAdd returns a + b mod q.
func fieldAdd(a, b fieldElement) fieldElement {
	x := uint32(a + b)
	return fieldReduceOnce(x)
}

// fieldSub returns a - b mod q.
func fieldSub(a, b fieldElement) fieldElement {
	x := uint32(a - b + q)
	return fieldReduceOnce(x)
}

// fieldMontgomeryMul returns a * b * R⁻¹ mod q.
func fieldMontgomeryMul(a, b fieldElement) fieldElement {
	x := uint64(a) * uint64(b)
	return fieldMontgomeryReduce(x)
}

// fieldMontgomeryReduce returns x * R⁻¹ mod q for x < q * R.
func fieldMontgomeryReduce(x uint64) fieldElement {
	t := uint32(x) * qNegInv
	u := (x + uint64(t)*q) >> 32
	return fieldReduceOnce(uint32(u))
}

// fieldMontgomeryMulSub returns a * (b - c). This operation is fused to save a
// fieldReduceOnce after the subtraction.
func fieldMontgomeryMulSub(a, b, c fieldElement) fieldElement {
	x := uint64(a) * uint64(b-c+q)
	return fieldMontgomeryReduce(x)
}

// fieldMontgomeryAddMul returns a * b + c * d. This operation is fused to save
// a fieldReduceOnce and a fieldReduce.
func fieldMontgomeryAddMul(a, b, c, d fieldElement) fieldElement {
	x := uint64(a) * uint64(b)
	x += uint64(c) * uint64(d)
	return fieldMontgomeryReduce(x)
}

const n = 256

// ringElement is a polynomial, an element of R_q.
type ringElement [n]fieldElement

// polyAdd adds two ringElements or nttElements.
func polyAdd[T ~[n]fieldElement](a, b T) (s T) {
	for i := range s {
		s[i] = fieldAdd(a[i], b[i])
	}
	return s
}

// polySub subtracts two ringElements or nttElements.
func polySub[T ~[n]fieldElement](a, b T) (s T) {
	for i := range s {
		s[i] = fieldSub(a[i], b[i])
	}
	return s
}

// nttElement is an NTT representation, an element of T_q.
type nttElement [n]fieldElement

// zetas are the values ζ^BitRev₈(k) mod q for each index k, converted to the
// Montgomery domain.
var zetas = [256]fieldElement{4193792, 25847, 5771523, 7861508, 237124, 7602457, 7504169, 466468, 1826347, 2353451, 8021166, 6288512, 3119733, 5495562, 3111497, 2680103, 2725464, 1024112, 7300517, 3585928, 7830929, 7260833, 2619752, 6271868, 6262231, 4520680, 6980856, 5102745, 1757237, 8360995, 4010497, 280005, 2706023, 95776, 3077325, 3530437, 6718724, 4788269, 5842901, 3915439, 4519302, 5336701, 3574422, 5512770, 3539968, 8079950, 2348700, 7841118, 6681150, 6736599, 3505694, 4558682, 3507263, 6239768, 6779997, 3699596, 811944, 531354, 954230, 3881043, 3900724, 5823537, 2071892, 5582638, 4450022, 6851714, 4702672, 5339162, 6927966, 3475950, 2176455, 6795196, 7122806, 1939314, 4296819, 7380215, 5190273, 5223087, 4747489, 126922, 3412210, 7396998, 2147896, 2715295, 5412772, 4686924, 7969390, 5903370, 7709315, 7151892, 8357436, 7072248, 7998430, 1349076, 1852771, 6949987, 5037034, 264944, 508951, 3097992, 44288, 7280319, 904516, 3958618, 4656075, 8371839, 1653064, 5130689, 2389356, 8169440, 759969, 7063561, 189548, 4827145, 3159746, 6529015, 5971092, 8202977, 1315589, 1341330, 1285669, 6795489, 7567685, 6940675, 5361315, 4499357, 4751448, 3839961, 2091667, 3407706, 2316500, 3817976, 5037939, 2244091, 5933984, 4817955, 266997, 2434439, 7144689, 3513181, 4860065, 4621053, 7183191, 5187039, 900702, 1859098, 909542, 819034, 495491, 6767243, 8337157, 7857917, 7725090, 5257975, 2031748, 3207046, 4823422, 7855319, 7611795, 4784579, 342297, 286988, 5942594, 4108315, 3437287, 5038140, 1735879, 203044, 2842341, 2691481, 5790267, 1265009, 4055324, 1247620, 2486353, 1595974, 4613401, 1250494, 2635921, 4832145, 5386378, 1869119, 1903435, 7329447, 7047359, 1237275, 5062207, 6950192, 7929317, 1312455, 3306115, 6417775, 7100756, 1917081, 5834105, 7005614, 1500165, 777191, 2235880, 3406031, 7838005, 5548557, 6709241, 6533464, 5796124, 4656147, 594136, 4603424, 6366809, 2432395, 2454455, 8215696, 1957272, 3369112, 185531, 7173032, 5196991, 162844, 1616392, 3014001, 810149, 1652634, 4686184, 6581310, 5341501, 3523897, 3866901, 269760, 2213111, 7404533, 1717735, 472078, 7953734, 1723600, 6577327, 1910376, 6712985, 7276084, 8119771, 4546524, 5441381, 6144432, 7959518, 6094090, 183443, 7403526, 1612842, 4834730, 7826001, 3919660, 8332111, 7018208, 3937738, 1400424, 7534263, 1976782}

// ntt maps a ringElement to its nttElement representation.
//
// It implements NTT, according to FIPS 203, Algorithm 9.
func ntt(f ringElement) nttElement {
	var m uint8
	for len := 128; len >= 1; len /= 2 {
		for start := 0; start < 256; start += 2 * len {
			m++
			zeta := zetas[m]
			// Bounds check elimination hint.
			f, flen := f[start:start+len], f[start+len:start+len+len]
			for j := 0; j < len; j++ {
				t := fieldMontgomeryMul(zeta, flen[j])
				flen[j] = fieldSub(f[j], t)
				f[j] = fieldAdd(f[j], t)
			}
		}
	}
	return nttElement(f)
}

// inverseNTT maps a nttElement back to the ringElement it represents.
//
// It implements NTT⁻¹, according to FIPS 203, Algorithm 10.
func inverseNTT(f nttElement) ringElement {
	var m uint8 = 255
	for len := 1; len < 256; len *= 2 {
		for start := 0; start < 256; start += 2 * len {
			zeta := zetas[m]
			m--
			// Bounds check elimination hint.
			f, flen := f[start:start+len], f[start+len:start+len+len]
			for j := 0; j < len; j++ {
				t := f[j]
				f[j] = fieldAdd(t, flen[j])
				// -z * (t - flen[j]) = z * (flen[j] - t)
				flen[j] = fieldMontgomeryMulSub(zeta, flen[j], t)
			}
		}
	}
	for i := range f {
		f[i] = fieldMontgomeryMul(f[i], 16382) // 16382 = 256⁻¹ * R mod q
	}
	return ringElement(f)
}

// nttMul multiplies two nttElements.
func nttMul(a, b nttElement) (p nttElement) {
	for i := range p {
		p[i] = fieldMontgomeryMul(a[i], b[i])
	}
	return p
}

// sampleNTT samples an nttElement uniformly at random from the seed rho and the
// indices s and r. It implements Step 3 of ExpandA, RejNTTPoly, and
// CoeffFromThreeBytes from FIPS 204, passing in ρ, s, and r instead of ρ'.
func sampleNTT(rho []byte, s, r byte) nttElement {
	G := sha3.NewShake128()
	G.Write(rho)
	G.Write([]byte{s, r})

	var a nttElement
	var j int         // index into a
	var buf [168]byte // buffered reads from B, matching the rate of SHAKE-128
	off := len(buf)   // index into buf, starts in a "buffer fully consumed" state
	for j < n {
		if off >= len(buf) {
			G.Read(buf[:])
			off = 0
		}
		v := uint32(buf[off]) | uint32(buf[off+1])<<8 | uint32(buf[off+2])<<16
		off += 3
		f, err := fieldToMontgomery(v & 0b01111111_11111111_11111111) // 23 bits
		if err != nil {
			continue
		}
		a[j] = f
		j++
	}
	return a
}

// sampleBoundedPoly samples a ringElement with coefficients in [−η, η] from the
// seed rho and the index r. It implements RejBoundedPoly and CoeffFromHalfByte
// from FIPS 204, passing in ρ and r separately from ExpandS.
func sampleBoundedPoly(rho []byte, r byte, p parameters) ringElement {
	H := sha3.NewShake256()
	H.Write(rho)
	H.Write([]byte{r, 0}) // IntegerToBytes(r, 2)

	var a ringElement
	var j int
	var buf [136]byte // buffered reads from H, matching the rate of SHAKE-256
	off := len(buf)   // index into buf, starts in a "buffer fully consumed" state
	for {
		if off >= len(buf) {
			H.Read(buf[:])
			off = 0
		}
		z0 := buf[off] & 0x0F
		z1 := buf[off] >> 4
		off++
		coeff, ok := coeffFromHalfByte(z0, p)
		if ok {
			a[j] = coeff
			j++
		}
		if j >= len(a) {
			break
		}
		coeff, ok = coeffFromHalfByte(z1, p)
		if ok {
			a[j] = coeff
			j++
		}
		if j >= len(a) {
			break
		}
	}
	return a
}

// sampleInBall samples a ringElement with coefficients in {−1, 0, 1}, and τ
// non-zero coefficients. It is not constant-time.
func sampleInBall(rho []byte, p parameters) ringElement {
	H := sha3.NewShake256()
	H.Write(rho)
	s := make([]byte, 8)
	H.Read(s)

	var c ringElement
	for i := 256 - p.τ; i < 256; i++ {
		j := make([]byte, 1)
		H.Read(j)
		for j[0] > byte(i) {
			H.Read(j)
		}
		c[i] = c[j[0]]
		// c[j] = (−1) ^ h[i+τ−256], where h are the bits in s in little-endian.
		// That is, -1⁰ = 1 if the bit is 0, -1¹ = -1 if it is 1.
		bitIdx := i + p.τ - 256
		bit := (s[bitIdx/8] >> (bitIdx % 8)) & 1
		if bit == 0 {
			c[j[0]] = one
		} else {
			c[j[0]] = minusOne
		}
	}

	return c
}

// coeffFromHalfByte implements CoeffFromHalfByte from FIPS 204.
//
// It maps a value in [0, 15] to a coefficient in [−η, η]
func coeffFromHalfByte(b byte, p parameters) (fieldElement, bool) {
	if b > 15 {
		panic("internal error: half-byte out of range")
	}
	switch p.η {
	case 2:
		// Return z = 2 − (b mod 5), which maps from
		//
		//     b = ( 14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0 )
		//
		// to
		//
		//   b%5 = (  4,  3,  2,  1,  0,  4,  3,  2,  1,  0,  4,  3,  2,  1,  0 )
		//
		// to
		//
		//     z = ( -2, -1,  0,  1,  2, -2, -1,  0,  1,  2, -2, -1,  0,  1,  2 )
		//
		if b > 14 {
			return 0, false
		}
		// Calculate b % 5 with Barrett reduction, to avoid a potentially
		// variable-time division.
		const barrettMultiplier = 0x3334 // ⌈2¹⁶ / 5⌉
		const barrettShift = 16          // log₂(2¹⁶)
		quotient := (uint32(b) * barrettMultiplier) >> barrettShift
		remainder := uint32(b) - quotient*5
		return fieldSubToMontgomery(2, remainder), true
	case 4:
		// Return z = 4 − b, which maps from
		//
		//   b = (  8,  7,  6,  5,  4,  3,  2,  1,  0 )
		//
		// to
		//
		//   z = ( −4, -3, -2, -1,  0,  1,  2,  3,  4 )
		//
		if b > 8 {
			return 0, false
		}
		return fieldSubToMontgomery(4, uint32(b)), true
	default:
		panic("internal error: unsupported η")
	}
}

// power2Round implements Power2Round from FIPS 204.
//
// It separates the bottom d = 13 bits of each 23-bit coefficient, rounding the
// high part based on the low part, and correcting the low part accordingly.
func power2Round(r fieldElement) (hi uint16, lo fieldElement) {
	rr := fieldFromMontgomery(r)
	// Add 2¹² - 1 to round up r1 by one if r0 > 2¹².
	// r is at most 2²³ - 2¹³ + 1, so rr + (2¹² - 1) won't overflow 23 bits.
	r1 := rr + 1<<12 - 1
	r1 >>= 13
	// r1 <= 2¹⁰ - 1
	// r1 * 2¹³ <= (2¹⁰ - 1) * 2¹³ = 2²³ - 2¹³ < q
	r0 := fieldSubToMontgomery(rr, r1<<13)
	return uint16(r1), r0
}

// highBits implements HighBits from FIPS 204.
func highBits(r ringElement, p parameters) [n]byte {
	var w [n]byte
	switch p.γ2 {
	case 32:
		for i := range n {
			w[i] = highBits32(fieldFromMontgomery(r[i]))
		}
	case 88:
		for i := range n {
			w[i] = highBits88(fieldFromMontgomery(r[i]))
		}
	default:
		panic("mldsa: internal error: unsupported γ2")
	}
	return w
}

// useHint implements UseHint from FIPS 204.
//
// It is not constant-time.
func useHint(r ringElement, h [n]byte, p parameters) [n]byte {
	var w [n]byte
	switch p.γ2 {
	case 32:
		for i := range n {
			w[i] = useHint32(r[i], h[i])
		}
	case 88:
		for i := range n {
			w[i] = useHint88(r[i], h[i])
		}
	default:
		panic("mldsa: internal error: unsupported γ2")
	}
	return w
}

// makeHint implements MakeHint from FIPS 204.
func makeHint(ct0, w, cs2 ringElement, p parameters) (h [n]byte, count1s int) {
	switch p.γ2 {
	case 32:
		for i := range n {
			h[i] = makeHint32(ct0[i], w[i], cs2[i])
			count1s += int(h[i])
		}
	case 88:
		for i := range n {
			h[i] = makeHint88(ct0[i], w[i], cs2[i])
			count1s += int(h[i])
		}
	default:
		panic("mldsa: internal error: unsupported γ2")
	}
	return h, count1s
}

// highBits32 implements HighBits from FIPS 204 for γ2 = (q - 1) / 32.
func highBits32(x uint32) byte {
	// The implementation is based on the reference implementation and on
	// BoringSSL. There are exhaustive tests in TestDecompose that compare it to
	// a straightforward implementation of Decompose from the spec, so for our
	// purposes it only has to work and be constant-time.
	r1 := (x + 127) >> 7
	r1 = (r1*1025 + (1 << 21)) >> 22
	r1 &= 0b1111
	return byte(r1)
}

// decompose32 implements Decompose from FIPS 204 for γ2 = (q - 1) / 32.
//
// r1 is in [0, 15].
func decompose32(r fieldElement) (r1 byte, r0 int32) {
	x := fieldFromMontgomery(r)
	r1 = highBits32(x)

	// r - r1 * (2 * γ2) mod± q
	r0 = int32(x) - int32(r1)*2*(q-1)/32
	r0 = constantTimeSelectLessOrEqual(q/2+1, r0, r0-q, r0)

	return r1, r0
}

// useHint32 implements UseHint from FIPS 204 for γ2 = (q - 1) / 32.
func useHint32(r fieldElement, hint byte) byte {
	const m = 16 // (q − 1) / (2 * γ2)
	r1, r0 := decompose32(r)
	if hint == 1 {
		if r0 > 0 {
			r1 = (r1 + 1) % m
		} else {
			// Underflow is safe, because it operates modulo 256 (since the type
			// is byte), which is a multiple of m.
			r1 = (r1 - 1) % m
		}
	}
	return r1
}

// makeHint32 implements MakeHint from FIPS 204 for γ2 = (q - 1) / 32.
func makeHint32(ct0, w, cs2 fieldElement) byte {
	// v1 = HighBits(r + z) = HighBits(w - cs2 + ct0 - ct0) = HighBits(w - cs2)
	rPlusZ := fieldSub(w, cs2)
	v1 := highBits32(fieldFromMontgomery(rPlusZ))
	// r1 = HighBits(r) = HighBits(w - cs2 + ct0)
	r1 := highBits32(fieldFromMontgomery(fieldAdd(rPlusZ, ct0)))

	return byte(constanttime.ByteEq(v1, r1) ^ 1)
}

// highBits88 implements HighBits from FIPS 204 for γ2 = (q - 1) / 88.
func highBits88(x uint32) byte {
	// Like highBits32, this is exhaustively tested in TestDecompose.
	r1 := (x + 127) >> 7
	r1 = (r1*11275 + (1 << 23)) >> 24
	r1 = constantTimeSelectEqual(r1, 44, 0, r1)
	return byte(r1)
}

// decompose88 implements Decompose from FIPS 204 for γ2 = (q - 1) / 88.
//
// r1 is in [0, 43].
func decompose88(r fieldElement) (r1 byte, r0 int32) {
	x := fieldFromMontgomery(r)
	r1 = highBits88(x)

	// r - r1 * (2 * γ2) mod± q
	r0 = int32(x) - int32(r1)*2*(q-1)/88
	r0 = constantTimeSelectLessOrEqual(q/2+1, r0, r0-q, r0)

	return r1, r0
}

// useHint88 implements UseHint from FIPS 204 for γ2 = (q - 1) / 88.
func useHint88(r fieldElement, hint byte) byte {
	const m = 44 // (q − 1) / (2 * γ2)
	r1, r0 := decompose88(r)
	if hint == 1 {
		if r0 > 0 {
			// (r1 + 1) mod m, for r1 in [0, m-1]
			if r1 == m-1 {
				r1 = 0
			} else {
				r1++
			}
		} else {
			// (r1 - 1) % m, for r1 in [0, m-1]
			if r1 == 0 {
				r1 = m - 1
			} else {
				r1--
			}
		}
	}
	return r1
}

// makeHint88 implements MakeHint from FIPS 204 for γ2 = (q - 1) / 88.
func makeHint88(ct0, w, cs2 fieldElement) byte {
	// Same as makeHint32 above.
	rPlusZ := fieldSub(w, cs2)
	v1 := highBits88(fieldFromMontgomery(rPlusZ))
	r1 := highBits88(fieldFromMontgomery(fieldAdd(rPlusZ, ct0)))
	return byte(constanttime.ByteEq(v1, r1) ^ 1)
}

// bitPack implements BitPack(r mod± q, γ₁-1, γ₁), which packs the centered
// coefficients of r into little-endian γ1+1-bit chunks. It appends to buf.
//
// It must only be applied to r with coefficients in [−γ₁+1, γ₁], as
// guaranteed by the rejection conditions in Sign.
func bitPack(b []byte, r ringElement, p parameters) []byte {
	switch p.γ1 {
	case 17:
		return bitPack18(b, r)
	case 19:
		return bitPack20(b, r)
	default:
		panic("mldsa: internal error: unsupported γ1")
	}
}

// bitPack18 implements BitPack(r mod± q, 2¹⁷-1, 2¹⁷), which packs the centered
// coefficients of r into little-endian 18-bit chunks. It appends to buf.
//
// It must only be applied to r with coefficients in [−2¹⁷+1, 2¹⁷], as
// guaranteed by the rejection conditions in Sign.
func bitPack18(buf []byte, r ringElement) []byte {
	out, v := sliceForAppend(buf, 18*n/8)
	const b = 1 << 17
	for i := 0; i < n; i += 4 {
		// b - [−2¹⁷+1, 2¹⁷] = [0, 2²⁸-1]
		w0 := b - fieldCenteredMod(r[i])
		v[0] = byte(w0 << 0)
		v[1] = byte(w0 >> 8)
		v[2] = byte(w0 >> 16)
		w1 := b - fieldCenteredMod(r[i+1])
		v[2] |= byte(w1 << 2)
		v[3] = byte(w1 >> 6)
		v[4] = byte(w1 >> 14)
		w2 := b - fieldCenteredMod(r[i+2])
		v[4] |= byte(w2 << 4)
		v[5] = byte(w2 >> 4)
		v[6] = byte(w2 >> 12)
		w3 := b - fieldCenteredMod(r[i+3])
		v[6] |= byte(w3 << 6)
		v[7] = byte(w3 >> 2)
		v[8] = byte(w3 >> 10)
		v = v[4*18/8:]
	}
	return out
}

// bitPack20 implements BitPack(r mod± q, 2¹⁹-1, 2¹⁹), which packs the centered
// coefficients of r into little-endian 20-bit chunks. It appends to buf.
//
// It must only be applied to r with coefficients in [−2¹⁹+1, 2¹⁹], as
// guaranteed by the rejection conditions in Sign.
func bitPack20(buf []byte, r ringElement) []byte {
	out, v := sliceForAppend(buf, 20*n/8)
	const b = 1 << 19
	for i := 0; i < n; i += 2 {
		// b - [−2¹⁹+1, 2¹⁹] = [0, 2²⁰-1]
		w0 := b - fieldCenteredMod(r[i])
		v[0] = byte(w0 << 0)
		v[1] = byte(w0 >> 8)
		v[2] = byte(w0 >> 16)
		w1 := b - fieldCenteredMod(r[i+1])
		v[2] |= byte(w1 << 4)
		v[3] = byte(w1 >> 4)
		v[4] = byte(w1 >> 12)
		v = v[2*20/8:]
	}
	return out
}

// bitUnpack implements BitUnpack(v, 2^γ1-1, 2^γ1), which unpacks each γ1+1 bits
// in little-endian into a coefficient in [-2^γ1+1, 2^γ1].
func bitUnpack(v []byte, p parameters) ringElement {
	switch p.γ1 {
	case 17:
		return bitUnpack18(v)
	case 19:
		return bitUnpack20(v)
	default:
		panic("mldsa: internal error: unsupported γ1")
	}
}

// bitUnpack18 implements BitUnpack(v, 2¹⁷-1, 2¹⁷), which unpacks each 18 bits
// in little-endian into a coefficient in [-2¹⁷+1, 2¹⁷].
func bitUnpack18(v []byte) ringElement {
	if len(v) != 18*n/8 {
		panic("mldsa: internal error: invalid bitUnpack18 input length")
	}
	const b = 1 << 17
	const mask18 = 1<<18 - 1
	var r ringElement
	for i := 0; i < n; i += 4 {
		w0 := uint32(v[0]) | uint32(v[1])<<8 | uint32(v[2])<<16
		r[i+0] = fieldSubToMontgomery(b, w0&mask18)
		w1 := uint32(v[2])>>2 | uint32(v[3])<<6 | uint32(v[4])<<14
		r[i+1] = fieldSubToMontgomery(b, w1&mask18)
		w2 := uint32(v[4])>>4 | uint32(v[5])<<4 | uint32(v[6])<<12
		r[i+2] = fieldSubToMontgomery(b, w2&mask18)
		w3 := uint32(v[6])>>6 | uint32(v[7])<<2 | uint32(v[8])<<10
		r[i+3] = fieldSubToMontgomery(b, w3&mask18)
		v = v[4*18/8:]
	}
	return r
}

// bitUnpack20 implements BitUnpack(v, 2¹⁹-1, 2¹⁹), which unpacks each 20 bits
// in little-endian into a coefficient in [-2¹⁹+1, 2¹⁹].
func bitUnpack20(v []byte) ringElement {
	if len(v) != 20*n/8 {
		panic("mldsa: internal error: invalid bitUnpack20 input length")
	}
	const b = 1 << 19
	const mask20 = 1<<20 - 1
	var r ringElement
	for i := 0; i < n; i += 2 {
		w0 := uint32(v[0]) | uint32(v[1])<<8 | uint32(v[2])<<16
		r[i+0] = fieldSubToMontgomery(b, w0&mask20)
		w1 := uint32(v[2])>>4 | uint32(v[3])<<4 | uint32(v[4])<<12
		r[i+1] = fieldSubToMontgomery(b, w1&mask20)
		v = v[2*20/8:]
	}
	return r
}

// sliceForAppend takes a slice and a requested number of bytes. It returns a
// slice with the contents of the given slice followed by that many bytes and a
// second slice that aliases into it and contains only the extra bytes. If the
// original slice has sufficient capacity then no allocation is performed.
func sliceForAppend(in []byte, n int) (head, tail []byte) {
	if total := len(in) + n; cap(in) >= total {
		head = in[:total]
	} else {
		head = make([]byte, total)
		copy(head, in)
	}
	tail = head[len(in):]
	return
}

// constantTimeSelectLessOrEqual returns yes if a <= b, no otherwise, in constant time.
func constantTimeSelectLessOrEqual(a, b, yes, no int32) int32 {
	return int32(constanttime.Select(constanttime.LessOrEq(int(a), int(b)), int(yes), int(no)))
}

// constantTimeSelectEqual returns yes if a == b, no otherwise, in constant time.
func constantTimeSelectEqual(a, b, yes, no uint32) uint32 {
	return uint32(constanttime.Select(constanttime.Eq(int32(a), int32(b)), int(yes), int(no)))
}

// constantTimeAbs returns the absolute value of x in constant time.
func constantTimeAbs(x int32) uint32 {
	return uint32(constantTimeSelectLessOrEqual(0, x, x, -x))
}
