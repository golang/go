// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package mldsa

import (
	"math/big"
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
	for a := range uint32(2 * q) {
		got := fieldReduceOnce(a)
		var exp uint32
		if a < q {
			exp = a
		} else {
			exp = a - q
		}
		if uint32(got) != exp {
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

func TestFieldMulSub(t *testing.T) {
	for _, a := range interestingValues() {
		for _, b := range interestingValues() {
			for _, c := range interestingValues() {
				got := fieldFromMontgomery(fieldMontgomeryMulSub(a.m, b.m, c.m))
				exp := uint32((uint64(a.v) * (uint64(b.v) + q - uint64(c.v))) % q)
				if got != exp {
					t.Fatalf("%d * (%d - %d) = %d, expected %d", a.v, b.v, c.v, got, exp)
				}
			}
		}
	}
}

func TestFieldAddMul(t *testing.T) {
	for _, a := range interestingValues() {
		for _, b := range interestingValues() {
			for _, c := range interestingValues() {
				for _, d := range interestingValues() {
					got := fieldFromMontgomery(fieldMontgomeryAddMul(a.m, b.m, c.m, d.m))
					exp := uint32((uint64(a.v)*uint64(b.v) + uint64(c.v)*uint64(d.v)) % q)
					if got != exp {
						t.Fatalf("%d + %d * %d = %d, expected %d", a.v, b.v, c.v, got, exp)
					}
				}
			}
		}
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
