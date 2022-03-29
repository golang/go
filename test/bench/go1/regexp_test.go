// Copyright 2013 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package go1

import (
	"math/rand"
	"regexp"
	"testing"
)

// benchmark based on regexp/exec_test.go

var regexpText []byte

func makeRegexpText(n int) []byte {
	rand.Seed(0) // For reproducibility.
	if len(regexpText) >= n {
		return regexpText[:n]
	}
	regexpText = make([]byte, n)
	for i := range regexpText {
		if rand.Intn(30) == 0 {
			regexpText[i] = '\n'
		} else {
			regexpText[i] = byte(rand.Intn(0x7E+1-0x20) + 0x20)
		}
	}
	return regexpText
}

func benchmark(b *testing.B, re string, n int) {
	r := regexp.MustCompile(re)
	t := makeRegexpText(n)
	b.ResetTimer()
	b.SetBytes(int64(n))
	for i := 0; i < b.N; i++ {
		if r.Match(t) {
			b.Fatal("match!")
		}
	}
}

const (
	easy0  = "ABCDEFGHIJKLMNOPQRSTUVWXYZ$"
	easy1  = "A[AB]B[BC]C[CD]D[DE]E[EF]F[FG]G[GH]H[HI]I[IJ]J$"
	medium = "[XYZ]ABCDEFGHIJKLMNOPQRSTUVWXYZ$"
	hard   = "[ -~]*ABCDEFGHIJKLMNOPQRSTUVWXYZ$"
)

func BenchmarkRegexpMatchEasy0_32(b *testing.B)  { benchmark(b, easy0, 32<<0) }
func BenchmarkRegexpMatchEasy0_1K(b *testing.B)  { benchmark(b, easy0, 1<<10) }
func BenchmarkRegexpMatchEasy1_32(b *testing.B)  { benchmark(b, easy1, 32<<0) }
func BenchmarkRegexpMatchEasy1_1K(b *testing.B)  { benchmark(b, easy1, 1<<10) }
func BenchmarkRegexpMatchMedium_32(b *testing.B) { benchmark(b, medium, 32<<0) }
func BenchmarkRegexpMatchMedium_1K(b *testing.B) { benchmark(b, medium, 1<<10) }
func BenchmarkRegexpMatchHard_32(b *testing.B)   { benchmark(b, hard, 32<<0) }
func BenchmarkRegexpMatchHard_1K(b *testing.B)   { benchmark(b, hard, 1<<10) }
