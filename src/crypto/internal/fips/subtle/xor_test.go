// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package subtle_test

import (
	"bytes"
	. "crypto/internal/fips/subtle"
	"crypto/rand"
	"fmt"
	"io"
	"testing"
)

func TestXORBytes(t *testing.T) {
	for n := 1; n <= 1024; n++ {
		if n > 16 && testing.Short() {
			n += n >> 3
		}
		for alignP := 0; alignP < 8; alignP++ {
			for alignQ := 0; alignQ < 8; alignQ++ {
				for alignD := 0; alignD < 8; alignD++ {
					p := make([]byte, alignP+n, alignP+n+10)[alignP:]
					q := make([]byte, alignQ+n, alignQ+n+10)[alignQ:]
					if n&1 != 0 {
						p = p[:n]
					} else {
						q = q[:n]
					}
					if _, err := io.ReadFull(rand.Reader, p); err != nil {
						t.Fatal(err)
					}
					if _, err := io.ReadFull(rand.Reader, q); err != nil {
						t.Fatal(err)
					}

					d := make([]byte, alignD+n, alignD+n+10)
					for i := range d {
						d[i] = 0xdd
					}
					want := make([]byte, len(d), cap(d))
					copy(want[:cap(want)], d[:cap(d)])
					for i := 0; i < n; i++ {
						want[alignD+i] = p[i] ^ q[i]
					}

					if XORBytes(d[alignD:], p, q); !bytes.Equal(d, want) {
						t.Fatalf("n=%d alignP=%d alignQ=%d alignD=%d:\n\tp = %x\n\tq = %x\n\td = %x\n\twant %x\n", n, alignP, alignQ, alignD, p, q, d, want)
					}
				}
			}
		}
	}
}

func TestXorBytesPanic(t *testing.T) {
	mustPanic(t, "subtle.XORBytes: dst too short", func() {
		XORBytes(nil, make([]byte, 1), make([]byte, 1))
	})
	mustPanic(t, "subtle.XORBytes: dst too short", func() {
		XORBytes(make([]byte, 1), make([]byte, 2), make([]byte, 3))
	})
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
				XORBytes(dst, s0, s1)
			}
		})
	}
}

func mustPanic(t *testing.T, expected string, f func()) {
	t.Helper()
	defer func() {
		switch msg := recover().(type) {
		case nil:
			t.Errorf("expected panic(%q), but did not panic", expected)
		case string:
			if msg != expected {
				t.Errorf("expected panic(%q), but got panic(%q)", expected, msg)
			}
		default:
			t.Errorf("expected panic(%q), but got panic(%T%v)", expected, msg, msg)
		}
	}()
	f()
}
