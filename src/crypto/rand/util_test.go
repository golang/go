// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand_test

import (
	"crypto/rand"
	"math/big"
	"testing"
)

// https://golang.org/issue/6849.
func TestPrimeSmall(t *testing.T) {
	for n := 2; n < 10; n++ {
		p, err := rand.Prime(rand.Reader, n)
		if err != nil {
			t.Fatalf("Can't generate %d-bit prime: %v", n, err)
		}
		if p.BitLen() != n {
			t.Fatalf("%v is not %d-bit", p, n)
		}
		if !p.ProbablyPrime(32) {
			t.Fatalf("%v is not prime", p)
		}
	}
}

// Test that passing bits < 2 causes Prime to return nil, error
func TestPrimeBitsLt2(t *testing.T) {
	if p, err := rand.Prime(rand.Reader, 1); p != nil || err == nil {
		t.Errorf("Prime should return nil, error when called with bits < 2")
	}
}

func TestInt(t *testing.T) {
	// start at 128 so the case of (max.BitLen() % 8) == 0 is covered
	for n := 128; n < 140; n++ {
		b := new(big.Int).SetInt64(int64(n))
		if i, err := rand.Int(rand.Reader, b); err != nil {
			t.Fatalf("Can't generate random value: %v, %v", i, err)
		}
	}
}

func testIntPanics(t *testing.T, b *big.Int) {
	defer func() {
		if err := recover(); err == nil {
			t.Errorf("Int should panic when called with max <= 0: %v", b)
		}
	}()
	rand.Int(rand.Reader, b)
}

// Test that passing a new big.Int as max causes Int to panic
func TestIntEmptyMaxPanics(t *testing.T) {
	b := new(big.Int)
	testIntPanics(t, b)
}

// Test that passing a negative value as max causes Int to panic
func TestIntNegativeMaxPanics(t *testing.T) {
	b := new(big.Int).SetInt64(int64(-1))
	testIntPanics(t, b)
}
