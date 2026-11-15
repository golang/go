// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"testing"

	gotoken "go/token"
)

func upstreamKeywordSet() map[string]bool {
	out := map[string]bool{}
	for i := 1; i < 512; i++ {
		tok := gotoken.Token(i)
		if tok.IsKeyword() {
			out[tok.String()] = true
		}
	}
	return out
}

func TestLogosKeywordTableMatchesUpstream(t *testing.T) {
	want := upstreamKeywordSet()
	var words []string
	for _, tok := range keywordMap {
		if tok != 0 {
			words = append(words, tok.String())
		}
	}
	if len(words) != len(want) {
		t.Errorf("keyword table holds %d words, want %d upstream keywords", len(words), len(want))
	}
	seen := map[string]bool{}
	for _, w := range words {
		if seen[w] {
			t.Errorf("keyword table holds %q twice", w)
		}
		seen[w] = true
		if !want[w] {
			t.Errorf("keyword table holds %q, which go/token does not list as a keyword", w)
		}
	}
	for w := range want {
		if !seen[w] {
			t.Errorf("keyword table lacks upstream keyword %q", w)
		}
	}
	for _, w := range words {
		checkWordKind(t, w)
	}
}

func TestLogosFourLetterWordsStayNames(t *testing.T) {
	const total = 26 * 26 * 26 * 26
	var buf [4]byte
	for n := range total {
		m := n
		for i := 3; i >= 0; i-- {
			buf[i] = byte('a' + m%26)
			m /= 26
		}
		checkWordKind(t, string(buf[:]))
	}
}
