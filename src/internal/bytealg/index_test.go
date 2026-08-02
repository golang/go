// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build amd64 || arm64 || loong64 || s390x || ppc64le || ppc64

package bytealg_test

import (
	"fmt"
	"internal/bytealg"
	"math/rand"
	"strings"
	"testing"
)

// indexRef is a reference implementation of Index, used as the
// correctness oracle for the assembly implementations.
func indexRef(hay, needle string) int {
	for i := 0; i+len(needle) <= len(hay); i++ {
		if hay[i:i+len(needle)] == needle {
			return i
		}
	}
	return -1
}

// check compares both Index and IndexString against the reference
// implementation for one input pair.
func check(t *testing.T, hay, needle string) {
	t.Helper()
	want := indexRef(hay, needle)
	if got := bytealg.IndexString(hay, needle); got != want {
		t.Errorf("IndexString(%q, %q) = %d, want %d", hay, needle, got, want)
	}
	if got := bytealg.Index([]byte(hay), []byte(needle)); got != want {
		t.Errorf("Index(%q, %q) = %d, want %d", hay, needle, got, want)
	}
}

// TestIndexExhaustive sweeps all needle lengths and all match positions
// over a range of haystack lengths chosen to cover both the scalar and
// SIMD paths and the boundary between them, including haystack lengths
// that are not a multiple of the vector width and matches inside the
// final (overlapping) block.
func TestIndexExhaustive(t *testing.T) {
	if bytealg.MaxLen < 2 {
		t.Skip("Index not enabled")
	}
	maxNeedle := bytealg.MaxLen
	if maxNeedle > 64 {
		maxNeedle = 64 // keep the sweep bounded
	}
	// Distinct-ish byte content so partial overlaps of the needle
	// never accidentally match: position i holds byte 'A'+i%23.
	content := func(n int) string {
		s := make([]byte, n)
		for i := range s {
			s[i] = byte('A' + i%23)
		}
		return string(s)
	}
	for n := 2; n <= maxNeedle; n++ {
		for _, extra := range []int{0, 1, 7, 13, 14, 15, 16, 17, 30, 31, 32, 33, 47, 48, 63, 64, 100} {
			h := n + extra
			hay := content(h)
			// A needle taken from every position must be found there:
			// earlier positions cannot match because every window of
			// content is unique.
			for pos := 0; pos+n <= h; pos++ {
				check(t, hay, hay[pos:pos+n])
			}
			// Absent needles: right length, wrong content.
			check(t, hay, strings.Repeat("z", n))
			// Wrong middle byte: anchors (first/last byte) match at
			// one position but the needle differs inside.
			corrupt := []byte(hay[h-n : h])
			corrupt[n/2] ^= 0x80
			check(t, hay, string(corrupt))
		}
	}
}

// TestIndexPathological exercises inputs that generate many anchor
// false positives per 16-byte block, stressing the candidate
// verification and mask-clearing paths.
func TestIndexPathological(t *testing.T) {
	if bytealg.MaxLen < 2 {
		t.Skip("Index not enabled")
	}
	tests := []struct {
		name string
		a, b string
	}{
		// Both anchors match at every even offset; middles never match.
		{"anchor false positives", strings.Repeat("ab", 100), "a" + strings.Repeat("c", 10) + "b"},
		// Every byte matches both anchors; verification fails everywhere.
		{"all candidates no match", strings.Repeat("a", 200), "a" + strings.Repeat("a", 10) + "b"},
		{"all candidates match at 0", strings.Repeat("a", 200), strings.Repeat("a", 12)},
		{"all candidates match late", strings.Repeat("a", 100) + "b" + strings.Repeat("a", 20), strings.Repeat("a", 8) + "b"},
		// Overlapping self-similar needle.
		{"overlapping periodic", strings.Repeat("abcab", 40), "abcabcab"},
		// Needle lengths 4..8 with the needle's own first/last byte
		// recurring throughout the haystack: forces many anchor
		// collisions per 16-byte block for every length in the {4-8}
		// class, exercising both the post-refine single-check verify
		// (multiple candidates) and its max(n-6,0) offset boundary at
		// each length, and the lone-candidate two-chunk verify.
		{"anchor collisions n=4 no match", strings.Repeat("a", 200), "aqqa"},
		{"anchor collisions n=5 no match", strings.Repeat("a", 200), "aqqqa"},
		{"anchor collisions n=6 no match", strings.Repeat("a", 200), "aqqqqa"},
		{"anchor collisions n=7 no match", strings.Repeat("a", 200), "aqqqqqa"},
		{"anchor collisions n=8 no match", strings.Repeat("a", 200), "aqqqqqqa"},
		{"anchor collisions n=4 match late", strings.Repeat("a", 150) + "aqqa" + strings.Repeat("a", 46), "aqqa"},
		{"anchor collisions n=5 match late", strings.Repeat("a", 150) + "aqqqa" + strings.Repeat("a", 45), "aqqqa"},
		{"anchor collisions n=6 match late", strings.Repeat("a", 150) + "aqqqqa" + strings.Repeat("a", 44), "aqqqqa"},
		{"anchor collisions n=7 match late", strings.Repeat("a", 150) + "aqqqqqa" + strings.Repeat("a", 43), "aqqqqqa"},
		{"anchor collisions n=8 match late", strings.Repeat("a", 150) + "aqqqqqqa" + strings.Repeat("a", 42), "aqqqqqqa"},
		// Match only at the very last valid offset.
		{"match at end", strings.Repeat("x", 50) + "needle", "needle"},
		{"match at start", "needle" + strings.Repeat("x", 50), "needle"},
		// Needle almost at the end, misaligned with the final block.
		{"near end miss", strings.Repeat("x", 53) + "needl!", "needle"},
		// Second anchor differs only at the last byte of the haystack.
		{"last byte differs", strings.Repeat("x", 60) + "ne", "nz"},
		{"last byte matches", strings.Repeat("x", 60) + "ne", "ne"},
	}
	// Needles whose first two and last two bytes all match at every
	// offset, so candidates pass both the anchor and refine checks and
	// only full verification can reject them: the one differing byte
	// sits at each end of the region the verify chunks must cover.
	for n := 5; n <= 32; n++ {
		for _, mid := range []int{2, n / 2, n - 3} {
			needle := []byte(strings.Repeat("a", n))
			needle[mid] = 'b'
			tests = append(tests,
				struct{ name, a, b string }{
					fmt.Sprintf("refine passes n=%d b[%d] no match", n, mid),
					strings.Repeat("a", 200), string(needle),
				},
				struct{ name, a, b string }{
					fmt.Sprintf("refine passes n=%d b[%d] match late", n, mid),
					strings.Repeat("a", 150) + string(needle) + strings.Repeat("a", 30), string(needle),
				})
		}
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			if len(tt.b) > bytealg.MaxLen {
				t.Skip("needle longer than MaxLen")
			}
			check(t, tt.a, tt.b)
		})
	}
}

// TestIndexRandom cross-checks the assembly implementations against the
// reference on random inputs over a small alphabet (to generate plenty
// of partial matches and anchor collisions).
func TestIndexRandom(t *testing.T) {
	if bytealg.MaxLen < 2 {
		t.Skip("Index not enabled")
	}
	r := rand.New(rand.NewSource(1))
	for _, alphabet := range []string{"a", "ab", "abc", "abcdefgh"} {
		for i := 0; i < 5000; i++ {
			n := 2 + r.Intn(bytealg.MaxLen-1)
			h := n + r.Intn(300)
			hay := make([]byte, h)
			for j := range hay {
				hay[j] = alphabet[r.Intn(len(alphabet))]
			}
			var needle string
			if r.Intn(2) == 0 && h > n {
				// Slice of the haystack: guaranteed to occur.
				pos := r.Intn(h - n)
				needle = string(hay[pos : pos+n])
			} else {
				nb := make([]byte, n)
				for j := range nb {
					nb[j] = alphabet[r.Intn(len(alphabet))]
				}
				needle = string(nb)
			}
			check(t, string(hay), needle)
		}
	}
}

// BenchmarkIndexString measures IndexString directly (bypassing the
// strings/bytes dispatch layers) across needle lengths and haystack
// sizes, so implementation thresholds can be recalibrated on future
// hardware.
//
// "text" searches random lowercase text for an uppercase needle placed
// at the very end: a full scan with a realistic (low) rate of anchor
// false positives. "hard" searches an alternating haystack for an
// absent needle whose first and last bytes match at every other
// offset: maximum verification pressure.
func BenchmarkIndexString(b *testing.B) {
	r := rand.New(rand.NewSource(1))
	base := make([]byte, 4096)
	for i := range base {
		base[i] = 'a' + byte(r.Intn(26))
	}
	sizes := []int{16, 24, 32, 64, 512, 4096}
	for _, n := range []int{2, 4, 8, 16, 32} {
		if n > bytealg.MaxLen {
			continue
		}
		for _, h := range sizes {
			if h < n {
				continue
			}
			hay := make([]byte, h)
			copy(hay, base)
			needle := make([]byte, n)
			for i := range needle {
				needle[i] = 'A' + byte(r.Intn(26))
			}
			copy(hay[h-n:], needle)
			s, sep := string(hay), string(needle)
			b.Run(fmt.Sprintf("text/n=%d/h=%d", n, h), func(b *testing.B) {
				b.SetBytes(int64(h))
				for i := 0; i < b.N; i++ {
					if bytealg.IndexString(s, sep) != h-n {
						b.Fatal("wrong result")
					}
				}
			})
		}
	}
	for _, n := range []int{4, 8, 16, 32} {
		if n > bytealg.MaxLen {
			continue
		}
		for _, h := range sizes {
			if h < n {
				continue
			}
			s := strings.Repeat("ab", h/2)
			sep := "a" + strings.Repeat("c", n-2) + "b"
			b.Run(fmt.Sprintf("hard/n=%d/h=%d", n, h), func(b *testing.B) {
				b.SetBytes(int64(h))
				for i := 0; i < b.N; i++ {
					if bytealg.IndexString(s, sep) != -1 {
						b.Fatal("wrong result")
					}
				}
			})
		}
	}
}
