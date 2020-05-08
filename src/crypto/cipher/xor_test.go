// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package cipher_test

import (
	"bytes"
	"crypto/cipher"
	"crypto/rand"
	"fmt"
	"io"
	"testing"
)

func TestXOR(t *testing.T) {
	for j := 1; j <= 1024; j++ {
		if testing.Short() && j > 16 {
			break
		}
		for alignP := 0; alignP < 2; alignP++ {
			for alignQ := 0; alignQ < 2; alignQ++ {
				for alignD := 0; alignD < 2; alignD++ {
					p := make([]byte, j)[alignP:]
					q := make([]byte, j)[alignQ:]
					d1 := make([]byte, j+alignD)[alignD:]
					d2 := make([]byte, j+alignD)[alignD:]
					if _, err := io.ReadFull(rand.Reader, p); err != nil {
						t.Fatal(err)
					}
					if _, err := io.ReadFull(rand.Reader, q); err != nil {
						t.Fatal(err)
					}
					cipher.XorBytes(d1, p, q)
					n := min(p, q)
					for i := 0; i < n; i++ {
						d2[i] = p[i] ^ q[i]
					}
					if !bytes.Equal(d1, d2) {
						t.Logf("p: %#v", p)
						t.Logf("q: %#v", q)
						t.Logf("expect: %#v", d2)
						t.Logf("result: %#v", d1)
						t.Fatal("not equal")
					}
				}
			}
		}
	}
}

func min(a, b []byte) int {
	n := len(a)
	if len(b) < n {
		n = len(b)
	}
	return n
}

func BenchmarkXORBytes(b *testing.B) {
	dst := make([]byte, 1<<15)
	data0 := make([]byte, 1<<15)
	data1 := make([]byte, 1<<15)
	sizes := []int64{1 << 3, 1 << 7, 1 << 11, 1 << 15}
	for _, size := range sizes {
		b.Run(fmt.Sprintf("%dBytes", size), func(b *testing.B) {
			s0 := data0[:size]
			s1 := data1[:size]
			b.SetBytes(int64(size))
			for i := 0; i < b.N; i++ {
				cipher.XorBytes(dst, s0, s1)
			}
		})
	}
}
