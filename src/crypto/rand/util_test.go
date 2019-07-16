// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand_test

import (
	"bytes"
	"crypto/rand"
	"fmt"
	"io"
	"math/big"
	mathrand "math/rand"
	"testing"
	"time"
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

type countingReader struct {
	r io.Reader
	n int
}

func (r *countingReader) Read(p []byte) (n int, err error) {
	n, err = r.r.Read(p)
	r.n += n
	return n, err
}

// Test that Int reads only the necessary number of bytes from the reader for
// max at each bit length
func TestIntReads(t *testing.T) {
	for i := 0; i < 32; i++ {
		max := int64(1 << uint64(i))
		t.Run(fmt.Sprintf("max=%d", max), func(t *testing.T) {
			reader := &countingReader{r: rand.Reader}

			_, err := rand.Int(reader, big.NewInt(max))
			if err != nil {
				t.Fatalf("Can't generate random value: %d, %v", max, err)
			}
			expected := (i + 7) / 8
			if reader.n != expected {
				t.Errorf("Int(reader, %d) should read %d bytes, but it read: %d", max, expected, reader.n)
			}
		})
	}
}

// Test that Int does not mask out valid return values
func TestIntMask(t *testing.T) {
	for max := 1; max <= 256; max++ {
		t.Run(fmt.Sprintf("max=%d", max), func(t *testing.T) {
			for i := 0; i < max; i++ {
				if testing.Short() && i == 0 {
					i = max - 1
				}
				var b bytes.Buffer
				b.WriteByte(byte(i))
				n, err := rand.Int(&b, big.NewInt(int64(max)))
				if err != nil {
					t.Fatalf("Can't generate random value: %d, %v", max, err)
				}
				if n.Int64() != int64(i) {
					t.Errorf("Int(reader, %d) should have returned value of %d, but it returned: %v", max, i, n)
				}
			}
		})
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

func BenchmarkPrime(b *testing.B) {
	r := mathrand.New(mathrand.NewSource(time.Now().UnixNano()))
	for i := 0; i < b.N; i++ {
		rand.Prime(r, 1024)
	}
}
