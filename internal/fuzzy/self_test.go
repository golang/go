// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fuzzy_test

import (
	"testing"

	. "golang.org/x/tools/internal/fuzzy"
)

func BenchmarkSelf_Matcher(b *testing.B) {
	idents := collectIdentifiers(b)
	patterns := generatePatterns()

	for i := 0; i < b.N; i++ {
		for _, pattern := range patterns {
			sm := NewMatcher(pattern)
			for _, ident := range idents {
				_ = sm.Score(ident)
			}
		}
	}
}

func BenchmarkSelf_SymbolMatcher(b *testing.B) {
	idents := collectIdentifiers(b)
	patterns := generatePatterns()

	for i := 0; i < b.N; i++ {
		for _, pattern := range patterns {
			sm := NewSymbolMatcher(pattern)
			for _, ident := range idents {
				_, _ = sm.Match([]string{ident})
			}
		}
	}
}
