// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.simd

package hex

import (
	"bytes"
	"fmt"
	"strings"
	"testing"
)

type kernel struct {
	name           string
	encode, decode func(dst, src []byte) int
}

func simdKernels() []kernel {
	var ks []kernel
	if haveSIMD {
		ks = append(ks, kernel{"archsimd", encodeSIMD, decodeSIMD})
	}
	return append(ks, kernel{"portable", encodePortable, decodePortable})
}

func TestSIMDKernels(t *testing.T) {
	src := make([]byte, 300)
	for i := range src {
		src[i] = byte(i * 167)
	}
	enc := fmt.Sprintf("%x", src)
	for _, k := range simdKernels() {
		for n := range len(src) + 1 {
			dst := make([]byte, 2*n)
			i := k.encode(dst, src[:n])
			if n >= 64 && i == 0 {
				t.Fatalf("%s: encoded nothing of %d bytes", k.name, n)
			}
			Encode(dst[2*i:], src[i:n])
			if string(dst) != enc[:2*n] {
				t.Fatalf("%s: encoded %x as %q, want %q", k.name, src[:n], dst, enc[:2*n])
			}
			for _, s := range []string{enc[:2*n], strings.ToUpper(enc[:2*n])} {
				got := make([]byte, n)
				j := k.decode(got, []byte(s))
				if n >= 64 && j == 0 {
					t.Fatalf("%s: decoded nothing of %q", k.name, s)
				}
				m, err := Decode(got[j/2:], []byte(s[j:]))
				if err != nil || j/2+m != n || !bytes.Equal(got, src[:n]) {
					t.Fatalf("%s: decoded %q as %x, %v; want %x", k.name, s, got, err, src[:n])
				}
			}
		}

		valid := strings.Repeat("0123456789abcdefABCDEF", 6)[:128]
		want, _ := DecodeString(valid)
		for p := range len(valid) {
			for _, c := range []byte{0, ' ', '/', ':', '@', 'G', '`', 'g', 0x7f, 0x80, 0xc6, 0xff} {
				s := []byte(valid)
				s[p] = c
				dst := make([]byte, len(s)/2)
				j := k.decode(dst, s)
				m, err := Decode(dst[j/2:], s[j:])
				if j > p || j/2+m != p/2 || err != InvalidByteError(c) || !bytes.Equal(dst[:p/2], want[:p/2]) {
					t.Fatalf("%s: Decode(%q) = %d, %v; want %d, %v", k.name, s, j/2+m, err, p/2, InvalidByteError(c))
				}
			}
		}
	}
}

func BenchmarkSIMDKernels(b *testing.B) {
	for _, size := range []int{16, 64, 256, 4096, 131072} {
		src := bytes.Repeat([]byte{2, 3, 5, 7, 9, 11, 13, 17}, size/8)
		enc := []byte(strings.Repeat("2b744faa", size/8))
		dst := make([]byte, 2*size)
		for _, k := range simdKernels() {
			b.Run("Encode/"+sizeName(size)+"/"+k.name, func(b *testing.B) {
				b.SetBytes(int64(size))
				for b.Loop() {
					i := k.encode(dst, src)
					Encode(dst[2*i:], src[i:])
				}
			})
			if size < 32 {
				continue
			}
			b.Run("Decode/"+sizeName(size)+"/"+k.name, func(b *testing.B) {
				b.SetBytes(int64(size))
				for b.Loop() {
					j := k.decode(dst, enc)
					Decode(dst[j/2:], enc[j:])
				}
			})
		}
	}
}
