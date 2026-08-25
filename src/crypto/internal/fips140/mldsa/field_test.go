// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mldsa

import (
	"crypto/internal/fips140/sha3"
	cryptorand "crypto/rand"
	"encoding/hex"
	"fmt"
	"math/big"
	"math/rand/v2"
	"testing"
)

type interestingValue struct {
	v uint32
	m fieldElement
}

// q is large enough that we can't exhaustively test all q × q inputs, so when
// we have two inputs  we test [0, q) on one side and a set of interesting
// values on the other side.
func interestingValues() []interestingValue {
	if testing.Short() {
		return []interestingValue{{v: q - 1, m: minusOne}}
	}
	var values []interestingValue
	for _, v := range []uint32{
		0,
		1,
		2,
		3,
		q - 3,
		q - 2,
		q - 1,
		q / 2,
		(q + 1) / 2,
	} {
		m, _ := fieldToMontgomery(v)
		values = append(values, interestingValue{v: v, m: m})
		// Also test values that have an interesting Montgomery representation.
		values = append(values, interestingValue{
			v: fieldFromMontgomery(fieldElement(v)), m: fieldElement(v)})
	}
	return values
}

func TestToFromMontgomery(t *testing.T) {
	for a := range uint32(q) {
		m, err := fieldToMontgomery(a)
		if err != nil {
			t.Fatalf("fieldToMontgomery(%d) returned error: %v", a, err)
		}
		exp := fieldElement((uint64(a) * R) % q)
		if m != exp {
			t.Fatalf("fieldToMontgomery(%d) = %d, expected %d", a, m, exp)
		}
		got := fieldFromMontgomery(m)
		if got != a {
			t.Fatalf("fieldFromMontgomery(fieldToMontgomery(%d)) = %d, expected %d", a, got, a)
		}
	}
}

func TestFieldAdd(t *testing.T) {
	t.Parallel()
	for _, a := range interestingValues() {
		for b := range fieldElement(q) {
			got := fieldAdd(a.m, b)
			exp := (a.m + b) % q
			if got != exp {
				t.Fatalf("%d + %d = %d, expected %d", a, b, got, exp)
			}
		}
	}
}

func TestFieldSub(t *testing.T) {
	t.Parallel()
	for _, a := range interestingValues() {
		for b := range fieldElement(q) {
			got := fieldSub(a.m, b)
			exp := (a.m + q - b) % q
			if got != exp {
				t.Fatalf("%d - %d = %d, expected %d", a, b, got, exp)
			}
		}
	}
}

func TestFieldSubToMontgomery(t *testing.T) {
	t.Parallel()
	for _, a := range interestingValues() {
		for b := range uint32(q) {
			got := fieldSubToMontgomery(a.v, b)
			diff := (a.v + q - b) % q
			exp := fieldElement((uint64(diff) * R) % q)
			if got != exp {
				t.Fatalf("fieldSubToMontgomery(%d, %d) = %d, expected %d", a.v, b, got, exp)
			}
		}
	}
}

func TestFieldReduceOnce(t *testing.T) {
	t.Parallel()
	for a := range fieldElementPartiallyReduced(2 * q) {
		got := fieldReduceOnce(a)
		var exp fieldElementPartiallyReduced
		if a < q {
			exp = a
		} else {
			exp = a - q
		}
		if fieldElementPartiallyReduced(got) != exp {
			t.Fatalf("fieldReduceOnce(%d) = %d, expected %d", a, got, exp)
		}
	}
}

func TestFieldMul(t *testing.T) {
	t.Parallel()
	for _, a := range interestingValues() {
		for b := range fieldElement(q) {
			got := fieldFromMontgomery(fieldMontgomeryMul(a.m, b))
			exp := uint32((uint64(a.v) * uint64(fieldFromMontgomery(b))) % q)
			if got != exp {
				t.Fatalf("%d * %d = %d, expected %d", a, b, got, exp)
			}
		}
	}
}

func TestFieldToMontgomeryOverflow(t *testing.T) {
	// fieldToMontgomery should reject inputs ≥ q.
	inputs := []uint32{
		q,
		q + 1,
		q + 2,
		1<<23 - 1,
		1 << 23,
		q + 1<<23,
		q + 1<<31,
		^uint32(0),
	}
	for _, in := range inputs {
		if _, err := fieldToMontgomery(in); err == nil {
			t.Fatalf("fieldToMontgomery(%d) did not return an error", in)
		}
	}
}

func TestFieldPartialReduce(t *testing.T) {
	t.Parallel()
	if testing.Short() {
		for _, a := range []uint32{
			0, 1, q - 1, q, 2*q - 1, 2 * q, 1<<23 - 1,
			1 << 23, 17 * q, 512 * q, ^uint32(0),
		} {
			got := fieldPartialReduce(a)
			if got >= 2*q {
				t.Fatalf("fieldPartialReduce(%d) = %d, out of the [0, 2q) range", a, got)
			}
			if uint32(got%q) != a%q || uint32(fieldReduceOnce(got)) != a%q {
				t.Fatalf("fieldPartialReduce(%d) = %d, which is not congruent to %d mod q", a, got, a)
			}
		}
		return
	}
	// Test the whole input space, tracking the expected value incrementally to
	// avoid a division in the loop.
	var a, exp uint32
	for {
		got := fieldPartialReduce(a)
		if got >= 2*q || uint32(fieldReduceOnce(got)) != exp {
			t.Fatalf("fieldPartialReduce(%d) = %d, expected a value congruent to %d mod q", a, got, exp)
		}
		exp++
		if exp == q {
			exp = 0
		}
		if a == 1<<32-1 {
			break
		}
		a++
	}
}

func TestFieldMontgomeryPartialReduce(t *testing.T) {
	t.Parallel()
	check := func(x uint64) {
		t.Helper()
		got := fieldMontgomeryPartialReduce(x)
		if got >= 2*q {
			t.Fatalf("fieldMontgomeryPartialReduce(%d) = %d, out of the [0, 2q) range", x, got)
		}
		// got ≡ x * R⁻¹ (mod q), so got * R ≡ x (mod q).
		if uint64(got)*(R%q)%q != x%q {
			t.Fatalf("fieldMontgomeryPartialReduce(%d) = %d, which is not congruent to %d * R⁻¹ mod q", x, got, x)
		}
	}
	// The largest input allowed by the contract, inclusive, along with the
	// bounds reached at the call sites and other notable values.
	for _, x := range []uint64{
		0, 1, q - 1, q, R - 1, R, 4 * q * q, 30 * q * q,
		32 * q * q, uint64(q-1) * (R - 1), 512 * q * 16382, q*R - 1, q * R,
	} {
		check(x)
	}
	for i := range uint64(1 << 20) {
		check(i * (q * R >> 20))
	}
}

func BitRev8(n uint8) uint8 {
	var r uint8
	r |= n >> 7 & 0b0000_0001
	r |= n >> 5 & 0b0000_0010
	r |= n >> 3 & 0b0000_0100
	r |= n >> 1 & 0b0000_1000
	r |= n << 1 & 0b0001_0000
	r |= n << 3 & 0b0010_0000
	r |= n << 5 & 0b0100_0000
	r |= n << 7 & 0b1000_0000
	return r
}

func CenteredMod(x, m uint32) int32 {
	x = x % m
	if x > m/2 {
		return int32(x) - int32(m)
	}
	return int32(x)
}

func reduceModQ(x int32) uint32 {
	x %= q
	if x < 0 {
		return uint32(x + q)
	}
	return uint32(x)
}

func TestCenteredMod(t *testing.T) {
	for x := range uint32(q * 2) {
		got := CenteredMod(uint32(x), q)
		if reduceModQ(got) != (x % q) {
			t.Fatalf("CenteredMod(%d) = %d, which is not congruent to %d mod %d", x, got, x, q)
		}
	}

	for x := range uint32(q) {
		r, _ := fieldToMontgomery(x)
		got := fieldCenteredMod(r)
		exp := CenteredMod(x, q)
		if got != exp {
			t.Fatalf("fieldCenteredMod(%d) = %d, expected %d", x, got, exp)
		}
	}
}

func TestInfinityNorm(t *testing.T) {
	for x := range uint32(q) {
		r, _ := fieldToMontgomery(x)
		got := fieldInfinityNorm(r)
		exp := CenteredMod(x, q)
		if exp < 0 {
			exp = -exp
		}
		if got != uint32(exp) {
			t.Fatalf("fieldInfinityNorm(%d) = %d, expected %d", x, got, exp)
		}
	}
}

func TestConstants(t *testing.T) {
	if fieldFromMontgomery(one) != 1 {
		t.Errorf("one constant incorrect")
	}
	if fieldFromMontgomery(minusOne) != q-1 {
		t.Errorf("minusOne constant incorrect")
	}
	if fieldInfinityNorm(one) != 1 {
		t.Errorf("one infinity norm incorrect")
	}
	if fieldInfinityNorm(minusOne) != 1 {
		t.Errorf("minusOne infinity norm incorrect")
	}

	if PublicKeySize44 != pubKeySize(params44) {
		t.Errorf("PublicKeySize44 constant incorrect")
	}
	if PublicKeySize65 != pubKeySize(params65) {
		t.Errorf("PublicKeySize65 constant incorrect")
	}
	if PublicKeySize87 != pubKeySize(params87) {
		t.Errorf("PublicKeySize87 constant incorrect")
	}
	if SignatureSize44 != sigSize(params44) {
		t.Errorf("SignatureSize44 constant incorrect")
	}
	if SignatureSize65 != sigSize(params65) {
		t.Errorf("SignatureSize65 constant incorrect")
	}
	if SignatureSize87 != sigSize(params87) {
		t.Errorf("SignatureSize87 constant incorrect")
	}
}

func TestPower2Round(t *testing.T) {
	t.Parallel()
	for x := range uint32(q) {
		rr, _ := fieldToMontgomery(x)
		t1, t0 := power2Round(rr)

		hi, err := fieldToMontgomery(uint32(t1) << 13)
		if err != nil {
			t.Fatalf("power2Round(%d): failed to convert high part to Montgomery: %v", x, err)
		}
		if r := fieldFromMontgomery(fieldAdd(hi, t0)); r != x {
			t.Fatalf("power2Round(%d) = (%d, %d), which reconstructs to %d, expected %d", x, t1, t0, r, x)
		}
	}
}

func SpecDecompose(rr fieldElement, p parameters) (R1 uint32, R0 int32) {
	r := fieldFromMontgomery(rr)
	if (q-1)%p.γ2 != 0 {
		panic("mldsa: internal error: unsupported denγ2")
	}
	γ2 := (q - 1) / uint32(p.γ2)
	r0 := CenteredMod(r, 2*γ2)
	diff := int32(r) - r0
	if diff == q-1 {
		r0 = r0 - 1
		return 0, r0
	} else {
		if diff < 0 || uint32(diff)%γ2 != 0 {
			panic("mldsa: internal error: invalid decomposition")
		}
		r1 := uint32(diff) / (2 * γ2)
		return r1, r0
	}
}

func TestDecompose(t *testing.T) {
	t.Run("ML-DSA-44", func(t *testing.T) {
		testDecompose(t, params44)
	})
	t.Run("ML-DSA-65,87", func(t *testing.T) {
		testDecompose(t, params65)
	})
}

func testDecompose(t *testing.T, p parameters) {
	t.Parallel()
	for x := range uint32(q) {
		rr, _ := fieldToMontgomery(x)
		r1, r0 := SpecDecompose(rr, p)

		// Check that SpecDecompose is correct.
		// r ≡ r1 * (2 * γ2) + r0 mod q
		γ2 := (q - 1) / uint32(p.γ2)
		reconstructed := reduceModQ(int32(r1*2*γ2) + r0)
		if reconstructed != x {
			t.Fatalf("SpecDecompose(%d) = (%d, %d), which reconstructs to %d, expected %d", x, r1, r0, reconstructed, x)
		}

		var gotR1 byte
		var gotR0 int32
		switch p.γ2 {
		case 88:
			gotR1, gotR0 = decompose88(rr)
			if gotR1 > 43 {
				t.Fatalf("decompose88(%d) returned r1 = %d, which is out of range", x, gotR1)
			}
		case 32:
			gotR1, gotR0 = decompose32(rr)
			if gotR1 > 15 {
				t.Fatalf("decompose32(%d) returned r1 = %d, which is out of range", x, gotR1)
			}
		default:
			t.Fatalf("unsupported denγ2: %d", p.γ2)
		}
		if uint32(gotR1) != r1 {
			t.Fatalf("highBits(%d) = %d, expected %d", x, gotR1, r1)
		}
		if gotR0 != r0 {
			t.Fatalf("lowBits(%d) = %d, expected %d", x, gotR0, r0)
		}
	}
}

func TestZetas(t *testing.T) {
	ζ := big.NewInt(1753)
	q := big.NewInt(q)
	for k, zeta := range zetas {
		// ζ^BitRev₈(k) mod q
		exp := new(big.Int).Exp(ζ, big.NewInt(int64(BitRev8(uint8(k)))), q)
		got := fieldFromMontgomery(zeta)
		if big.NewInt(int64(got)).Cmp(exp) != 0 {
			t.Errorf("zetas[%d] = %v, expected %v", k, got, exp)
		}
	}
}

func randomRingElement() ringElement {
	var f ringElement
	for i := range f {
		f[i] = fieldElement(rand.Uint32N(q))
	}
	return f
}

func randomNTTElement() nttElement {
	var f nttElement
	for i := range f {
		f[i] = fieldElementPartiallyReduced(rand.Uint32N(2 * q))
	}
	return f
}

func checkNTTElementBounds(t *testing.T, a nttElement) {
	t.Helper()
	for i, v := range a {
		if v >= 2*q {
			t.Fatalf("coefficient %d is %d, out of the [0, 2q) range", i, v)
		}
	}
}

func TestNTTRoundTrip(t *testing.T) {
	t.Parallel()
	check := func(f ringElement) {
		t.Helper()
		fHat := ntt(f)
		checkNTTElementBounds(t, fHat)
		if got := inverseNTT(fHat); got != f {
			t.Fatal("inverseNTT(ntt(f)) != f")
		}
	}
	var worst ringElement
	for i := range worst {
		worst[i] = q - 1
	}
	check(worst)
	check(ringElement{})
	for range 100 {
		check(randomRingElement())
	}
}

// TestInverseNTTLazyReduction checks that inverseNTT returns the same result
// for a partially reduced input and for the fully reduced input representing
// the same element. The all 2q - 1 input maximizes the intermediate values,
// which reach up to 512q, close to the uint32 limit.
func TestInverseNTTLazyReduction(t *testing.T) {
	t.Parallel()
	check := func(f nttElement) {
		t.Helper()
		var reduced nttElement
		for i := range f {
			reduced[i] = fieldElementPartiallyReduced(fieldReduceOnce(f[i]))
		}
		if inverseNTT(f) != inverseNTT(reduced) {
			t.Fatal("inverseNTT(f) != inverseNTT(f mod q)")
		}
	}
	var worst nttElement
	for i := range worst {
		worst[i] = 2*q - 1
	}
	check(worst)
	for range 100 {
		check(randomNTTElement())
	}
}

// TestNTTElementBounds checks that the functions producing nttElements keep
// their coefficients in the [0, 2q) range assumed across the package, in
// particular by the bound analysis of inverseNTT and the wide accumulators.
func TestNTTElementBounds(t *testing.T) {
	t.Parallel()
	var worstRing ringElement
	for i := range worstRing {
		worstRing[i] = q - 1
	}
	var worstNTT nttElement
	for i := range worstNTT {
		worstNTT[i] = 2*q - 1
	}

	checkNTTElementBounds(t, ntt(worstRing))
	checkNTTElementBounds(t, nttMul(worstNTT, worstNTT))

	var a nttElement
	buf := make([]byte, 5*168)
	cryptorand.Read(buf)
	parseSampleNTT(&a, 0, buf)
	checkNTTElementBounds(t, a)

	for range 100 {
		checkNTTElementBounds(t, ntt(randomRingElement()))
		checkNTTElementBounds(t, nttMul(randomNTTElement(), randomNTTElement()))
	}
}

// TestNTTAccumulator checks the wide accumulator functions against
// coefficient-wise modular arithmetic, in the same shapes used by key
// generation (a 2q² initial value and 4l * q² of products) and verification
// (4l * q² of products and a 4q² subtraction), with worst-case and random
// inputs.
func TestNTTAccumulator(t *testing.T) {
	t.Parallel()
	// std returns the standard domain value of a coefficient.
	std := func(v fieldElementPartiallyReduced) uint64 {
		return uint64(fieldFromMontgomery(fieldReduceOnce(v)))
	}
	check := func(init nttElement, a, b [maxL]nttElement, sub bool) {
		t.Helper()
		var p wideNTTElement
		nttWiden(&p, &init)
		for j := range maxL {
			nttMulAdd(&p, &a[j], &b[j])
		}
		if sub {
			nttMulSub(&p, &a[0], &b[0])
		}
		got := nttReduce(&p)
		checkNTTElementBounds(t, got)
		for i := range got {
			exp := std(init[i])
			for j := range maxL {
				exp = (exp + std(a[j][i])*std(b[j][i])) % q
			}
			if sub {
				exp = (exp + q - std(a[0][i])*std(b[0][i])%q) % q
			}
			if std(got[i]) != exp {
				t.Fatalf("accumulator coefficient %d is %d, expected %d", i, std(got[i]), exp)
			}
		}
	}

	var worst nttElement
	for i := range worst {
		worst[i] = 2*q - 1
	}
	var worstL [maxL]nttElement
	for i := range worstL {
		worstL[i] = worst
	}
	check(worst, worstL, worstL, false)
	check(nttElement{}, worstL, worstL, true)

	for range 10 {
		var a, b [maxL]nttElement
		for i := range a {
			a[i], b[i] = randomNTTElement(), randomNTTElement()
		}
		check(randomNTTElement(), a, b, false)
		check(nttElement{}, a, b, true)
	}
}

// parseSampleNTTReference is a direct transcription of the RejNTTPoly loop and
// CoeffFromThreeBytes from FIPS 204, sharing only the Montgomery conversion
// with parseSampleNTT so that representations can be compared exactly.
func parseSampleNTTReference(a *nttElement, j int, buf []byte) int {
	for len(buf) > 0 && j < n {
		z := uint32(buf[0]) | uint32(buf[1])<<8 | uint32(buf[2]&0x7f)<<16
		buf = buf[3:]
		if z < q {
			a[j] = fieldMontgomeryPartialMul(fieldElementPartiallyReduced(z), RR)
			j++
		}
	}
	return j
}

func TestParseSampleNTT(t *testing.T) {
	t.Parallel()
	// Candidates that exercise the acceptance boundary, including values with
	// the top bit of the third byte set, which must be ignored.
	candidates := []uint32{0, 1, q - 1, q, q + 1, 1<<23 - 1, q | 1<<23, 1<<24 - 1}
	randomBuf := func(size int) []byte {
		buf := make([]byte, 0, size)
		for len(buf) < size {
			v := candidates[rand.IntN(len(candidates))]
			buf = append(buf, byte(v), byte(v>>8), byte(v>>16))
		}
		return buf
	}
	check := func(j0 int, buf []byte) {
		t.Helper()
		var got, exp nttElement
		gotJ := parseSampleNTT(&got, j0, buf)
		expJ := parseSampleNTTReference(&exp, j0, buf)
		if gotJ != expJ {
			t.Fatalf("parseSampleNTT(%d, %x) = %d, expected %d", j0, buf, gotJ, expJ)
		}
		for i := j0; i < gotJ; i++ {
			if got[i] != exp[i] {
				t.Fatalf("parseSampleNTT(%d, %x): a[%d] = %d, expected %d", j0, buf, i, got[i], exp[i])
			}
		}
	}
	for range 10 {
		for _, size := range []int{3, 6, 9, 24, 168, 840} {
			for _, j0 := range []int{0, 100, 253, 254, 255, 256} {
				check(j0, randomBuf(size))
			}
		}
	}

	// An all-rejected buffer must not advance j.
	allRejected := make([]byte, 0, 840)
	for range 280 {
		v := uint32(q)
		allRejected = append(allRejected, byte(v), byte(v>>8), byte(v>>16))
	}
	var a nttElement
	if j := parseSampleNTT(&a, 0, allRejected); j != 0 {
		t.Fatalf("parseSampleNTT accepted %d rejected candidates", j)
	}

	// Resumed filling with one 168-byte block at a time, like computeMatrixA.
	var got, exp nttElement
	gotJ, expJ := 0, 0
	for range 100 {
		if gotJ >= n {
			break
		}
		buf := randomBuf(168)
		gotJ = parseSampleNTT(&got, gotJ, buf)
		expJ = parseSampleNTTReference(&exp, expJ, buf)
		if gotJ != expJ {
			t.Fatalf("resumed parseSampleNTT returned %d, expected %d", gotJ, expJ)
		}
	}
	if gotJ != n {
		t.Fatal("failed to fill the element in 100 blocks")
	}
	for i := range got {
		if got[i] != exp[i] {
			t.Fatalf("resumed parseSampleNTT: a[%d] = %d, expected %d", i, got[i], exp[i])
		}
	}
}

// TestAccumulated computes the hash of the following 12 values, as ASCII
// decimals with an optional leading - sign and separated by newlines, for all
// elements r in ℤq from 0 to q-1:
//
//   - r mod± q
//   - ‖r‖∞ = |r mod± q|
//   - r1, r0 = Power2Round(r)
//
// For ML-DSA-44 (γ₂ = (q - 1) / 88):
//   - HighBits(r) = UseHint(0, r)
//   - UseHint(1, r)
//   - LowBits(r)
//   - ‖LowBits(r)‖∞ = |LowBits(r)|
//
// For ML-DSA-65 and ML-DSA-87 (γ₂ = (q - 1) / 32):
//   - HighBits(r) = UseHint(0, r)
//   - UseHint(1, r)
//   - LowBits(r)
//   - ‖LowBits(r)‖∞ = |LowBits(r)|
//
// Note that HighBits(r), LowBits(r) = Decompose(r).
func TestAccumulated(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping accumulated test in short mode")
	}

	o := sha3.NewShake128()
	for x := range uint32(q) {
		r, _ := fieldToMontgomery(x)
		fmt.Fprintf(o, "%d\n", fieldCenteredMod(r))
		fmt.Fprintf(o, "%d\n", fieldInfinityNorm(r))

		hi, lo := power2Round(r)
		fmt.Fprintf(o, "%d\n", hi)
		fmt.Fprintf(o, "%d\n", fieldFromMontgomery(lo))

		r1, r0 := decompose88(r)
		if r1x := highBits88(fieldFromMontgomery(r)); r1x != r1 {
			t.Fatalf("highBits88(%d) = %d, expected %d", x, r1x, r1)
		}
		if r1h0 := useHint88(r, 0); r1h0 != r1 {
			t.Fatalf("useHint88(%d, 0) = %d, expected %d", x, r1h0, r1)
		}

		fmt.Fprintf(o, "%d\n", r1)
		fmt.Fprintf(o, "%d\n", useHint88(r, 1))
		fmt.Fprintf(o, "%d\n", r0)
		fmt.Fprintf(o, "%d\n", constantTimeAbs(r0))

		r1, r0 = decompose32(r)
		if r1x := highBits32(fieldFromMontgomery(r)); r1x != r1 {
			t.Fatalf("highBits32(%d) = %d, expected %d", x, r1x, r1)
		}
		if r1h0 := useHint32(r, 0); r1h0 != r1 {
			t.Fatalf("useHint32(%d, 0) = %d, expected %d", x, r1h0, r1)
		}

		fmt.Fprintf(o, "%d\n", r1)
		fmt.Fprintf(o, "%d\n", useHint32(r, 1))
		fmt.Fprintf(o, "%d\n", r0)
		fmt.Fprintf(o, "%d\n", constantTimeAbs(r0))
	}

	// The expected value is documented at https://c2sp.org/CCTV/ML-DSA, and
	// tested against https://github.com/FiloSottile/mldsa-py.
	expected := "f930663417278156ab05d940294a77210a809c924d8ab63ec72f4526247602c7"
	if got := hex.EncodeToString(o.Sum(nil)); got != expected {
		t.Errorf("got %s, expected %s", got, expected)
	}
}
