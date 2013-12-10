// Copyright 2013 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand_test

import (
	"crypto/rand"
	"testing"
)

// http://golang.org/issue/6849.
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
