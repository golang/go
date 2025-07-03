// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package rand_test

import (
	"crypto/rand"
	"fmt"
	"testing"
)

func TestText(t *testing.T) {
	set := make(map[string]struct{}) // hold every string produced
	var indexSet [26]map[rune]int    // hold every char produced at every position
	for i := range indexSet {
		indexSet[i] = make(map[rune]int)
	}

	// not getting a char in a position: (31/32)¹⁰⁰⁰ = 1.6e-14
	// test completion within 1000 rounds: (1-(31/32)¹⁰⁰⁰)²⁶ = 0.9999999999996
	// empirically, this should complete within 400 rounds = 0.999921
	rounds := 1000
	var done bool
	for range rounds {
		s := rand.Text()
		if len(s) != 26 {
			t.Errorf("len(Text()) = %d, want = 26", len(s))
		}
		for i, r := range s {
			if ('A' > r || r > 'Z') && ('2' > r || r > '7') {
				t.Errorf("Text()[%d] = %v, outside of base32 alphabet", i, r)
			}
		}
		if _, ok := set[s]; ok {
			t.Errorf("Text() = %s, duplicate of previously produced string", s)
		}
		set[s] = struct{}{}

		done = true
		for i, r := range s {
			indexSet[i][r]++
			if len(indexSet[i]) != 32 {
				done = false
			}
		}
		if done {
			break
		}
	}
	if !done {
		t.Errorf("failed to produce every char at every index after %d rounds", rounds)
		indexSetTable(t, indexSet)
	}
}

func indexSetTable(t *testing.T, indexSet [26]map[rune]int) {
	alphabet := "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"
	line := "   "
	for _, r := range alphabet {
		line += fmt.Sprintf(" %3s", string(r))
	}
	t.Log(line)
	for i, set := range indexSet {
		line = fmt.Sprintf("%2d:", i)
		for _, r := range alphabet {
			line += fmt.Sprintf(" %3d", set[r])
		}
		t.Log(line)
	}
}
