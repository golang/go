// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package subtle_test

import (
	"bytes"
	"crypto/rand"
	. "crypto/subtle"
	"fmt"
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
					p := make([]byte, alignP+n, alignP+n+100)[alignP:]
					q := make([]byte, alignQ+n, alignQ+n+100)[alignQ:]
					if n&1 != 0 {
						p = p[:n]
					} else {
						q = q[:n]
					}
					rand.Read(p)
					rand.Read(q)

					d := make([]byte, alignD+n+100)
					rand.Read(d)

					want := bytes.Clone(d)
					for i := range n {
						want[alignD+i] = p[i] ^ q[i]
					}

					if nn := XORBytes(d[alignD:], p, q); !bytes.Equal(d, want) {
						t.Errorf("n=%d alignP=%d alignQ=%d alignD=%d:\n\tp = %x\n\tq = %x\n\td = %x\n\twant %x\n", n, alignP, alignQ, alignD, p, q, d, want)
					} else if nn != n {
						t.Errorf("n=%d alignP=%d alignQ=%d alignD=%d: got %d, want %d", n, alignP, alignQ, alignD, nn, n)
					}
					p1 := bytes.Clone(p)
					if nn := XORBytes(p, p, q); !bytes.Equal(p, want[alignD:alignD+n]) {
						t.Errorf("n=%d alignP=%d alignQ=%d alignD=%d:\n\tp = %x\n\tq = %x\n\td = %x\n\twant %x\n", n, alignP, alignQ, alignD, p, q, d, want)
					} else if nn != n {
						t.Errorf("n=%d alignP=%d alignQ=%d alignD=%d: got %d, want %d", n, alignP, alignQ, alignD, nn, n)
					}
					if nn := XORBytes(q, p1, q); !bytes.Equal(q, want[alignD:alignD+n]) {
						t.Errorf("n=%d alignP=%d alignQ=%d alignD=%d:\n\tp = %x\n\tq = %x\n\td = %x\n\twant %x\n", n, alignP, alignQ, alignD, p, q, d, want)
					} else if nn != n {
						t.Errorf("n=%d alignP=%d alignQ=%d alignD=%d: got %d, want %d", n, alignP, alignQ, alignD, nn, n)
					}

					if nn := XORBytes(p, p, p); !bytes.Equal(p, make([]byte, n)) {
						t.Errorf("n=%d alignP=%d alignQ=%d alignD=%d: got %x, want %x", n, alignP, alignQ, alignD, p, make([]byte, n))
					} else if nn != n {
						t.Errorf("n=%d alignP=%d alignQ=%d alignD=%d: got %d, want %d", n, alignP, alignQ, alignD, nn, n)
					}
					if nn := XORBytes(p1, q, q); !bytes.Equal(p1, make([]byte, n)) {
						t.Errorf("n=%d alignP=%d alignQ=%d alignD=%d: got %x, want %x", n, alignP, alignQ, alignD, p1, make([]byte, n))
					} else if nn != n {
						t.Errorf("n=%d alignP=%d alignQ=%d alignD=%d: got %d, want %d", n, alignP, alignQ, alignD, nn, n)
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
	mustPanic(t, "subtle.XORBytes: invalid overlap", func() {
		x := make([]byte, 3)
		XORBytes(x, x[1:], make([]byte, 2))
	})
	mustPanic(t, "subtle.XORBytes: invalid overlap", func() {
		x := make([]byte, 3)
		XORBytes(x, make([]byte, 2), x[1:])
	})
}

func BenchmarkXORBytes(b *testing.B) {
	dst := make([]byte, 1<<15)
	data0 := make([]byte, 1<<15)
	data1 := make([]byte, 1<<15)
	sizes := []int64{1 << 3, 1 << 7, 1 << 11, 1 << 13, 1 << 15}
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

func BenchmarkXORBytesAlignment(b *testing.B) {
	dst := make([]byte, 8+1<<11)
	data0 := make([]byte, 8+1<<11)
	data1 := make([]byte, 8+1<<11)
	sizes := []int64{1 << 3, 1 << 7, 1 << 11}
	for _, size := range sizes {
		for offset := int64(0); offset < 8; offset++ {
			b.Run(fmt.Sprintf("%dBytes%dOffset", size, offset), func(b *testing.B) {
				d := dst[offset : offset+size]
				s0 := data0[offset : offset+size]
				s1 := data1[offset : offset+size]
				b.SetBytes(int64(size))
				for i := 0; i < b.N; i++ {
					XORBytes(d, s0, s1)
				}
			})
		}
	}
}

func mustPanic(t *testing.T, expected string, f func()) {
	t.Helper()
	defer func() {
		t.Helper()
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
