// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"testing"

	gotoken "go/token"
)

func TestLogosKeywordMapHoldsUpstreamOnly(t *testing.T) {
	occupied := 0
	for slot, tok := range keywordMap {
		if tok == 0 {
			continue
		}
		occupied++
		if !gotoken.IsKeyword(tok.String()) {
			t.Errorf("keywordMap slot %d holds token %q: not an upstream Go keyword", slot, tok.String())
		}
	}
	if occupied != 25 {
		t.Fatalf("keywordMap holds %d occupied slots, want the 25 upstream Go keywords", occupied)
	}
	for _, w := range []string{"errdefer", "yield", "scope", "group"} {
		if tok := keywordMap[hash([]byte(w))]; tok != 0 && tokStrFast(tok) == w {
			t.Errorf("keywordMap maps %q to keyword token %q; SPEC 4.1 keeps it contextual", w, tok.String())
		}
	}
}
