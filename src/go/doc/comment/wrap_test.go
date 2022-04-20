// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package comment

import (
	"flag"
	"fmt"
	"math/rand"
	"testing"
	"time"
	"unicode/utf8"
)

var wrapSeed = flag.Int64("wrapseed", 0, "use `seed` for wrap test (default auto-seeds)")

func TestWrap(t *testing.T) {
	if *wrapSeed == 0 {
		*wrapSeed = time.Now().UnixNano()
	}
	t.Logf("-wrapseed=%#x\n", *wrapSeed)
	r := rand.New(rand.NewSource(*wrapSeed))

	// Generate words of random length.
	s := "1234567890αβcdefghijklmnopqrstuvwxyz"
	sN := utf8.RuneCountInString(s)
	var words []string
	for i := 0; i < 100; i++ {
		n := 1 + r.Intn(sN-1)
		if n >= 12 {
			n++ // extra byte for β
		}
		if n >= 11 {
			n++ // extra byte for α
		}
		words = append(words, s[:n])
	}

	for n := 1; n <= len(words) && !t.Failed(); n++ {
		t.Run(fmt.Sprint("n=", n), func(t *testing.T) {
			words := words[:n]
			t.Logf("words: %v", words)
			for max := 1; max < 100 && !t.Failed(); max++ {
				t.Run(fmt.Sprint("max=", max), func(t *testing.T) {
					seq := wrap(words, max)

					// Compute score for seq.
					start := 0
					score := int64(0)
					if len(seq) == 0 {
						t.Fatalf("wrap seq is empty")
					}
					if seq[0] != 0 {
						t.Fatalf("wrap seq does not start with 0")
					}
					for _, n := range seq[1:] {
						if n <= start {
							t.Fatalf("wrap seq is non-increasing: %v", seq)
						}
						if n > len(words) {
							t.Fatalf("wrap seq contains %d > %d: %v", n, len(words), seq)
						}
						size := -1
						for _, s := range words[start:n] {
							size += 1 + utf8.RuneCountInString(s)
						}
						if n-start == 1 && size >= max {
							// no score
						} else if size > max {
							t.Fatalf("wrap used overlong line %d:%d: %v", start, n, words[start:n])
						} else if n != len(words) {
							score += int64(max-size)*int64(max-size) + wrapPenalty(words[n-1])
						}
						start = n
					}
					if start != len(words) {
						t.Fatalf("wrap seq does not use all words (%d < %d): %v", start, len(words), seq)
					}

					// Check that score matches slow reference implementation.
					slowSeq, slowScore := wrapSlow(words, max)
					if score != slowScore {
						t.Fatalf("wrap score = %d != wrapSlow score %d\nwrap: %v\nslow: %v", score, slowScore, seq, slowSeq)
					}
				})
			}
		})
	}
}

// wrapSlow is an O(n²) reference implementation for wrap.
// It returns a minimal-score sequence along with the score.
// It is OK if wrap returns a different sequence as long as that
// sequence has the same score.
func wrapSlow(words []string, max int) (seq []int, score int64) {
	// Quadratic dynamic programming algorithm for line wrapping problem.
	// best[i] tracks the best score possible for words[:i],
	// assuming that for i < len(words) the line breaks after those words.
	// bestleft[i] tracks the previous line break for best[i].
	best := make([]int64, len(words)+1)
	bestleft := make([]int, len(words)+1)
	best[0] = 0
	for i, w := range words {
		if utf8.RuneCountInString(w) >= max {
			// Overlong word must appear on line by itself. No effect on score.
			best[i+1] = best[i]
			continue
		}
		best[i+1] = 1e18
		p := wrapPenalty(w)
		n := -1
		for j := i; j >= 0; j-- {
			n += 1 + utf8.RuneCountInString(words[j])
			if n > max {
				break
			}
			line := int64(n-max)*int64(n-max) + p
			if i == len(words)-1 {
				line = 0 // no score for final line being too short
			}
			s := best[j] + line
			if best[i+1] > s {
				best[i+1] = s
				bestleft[i+1] = j
			}
		}
	}

	// Recover least weight sequence from bestleft.
	n := 1
	for m := len(words); m > 0; m = bestleft[m] {
		n++
	}
	seq = make([]int, n)
	for m := len(words); m > 0; m = bestleft[m] {
		n--
		seq[n] = m
	}
	return seq, best[len(words)]
}
