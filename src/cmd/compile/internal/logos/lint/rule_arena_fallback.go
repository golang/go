// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"cmp"
	"slices"
)

var arenaFallback = Rule{
	ID:    "perf::arena_fallback",
	Group: Perf,
	Go:    true,
}

type FallbackFinding struct {
	File      string
	Line, Col uint
	Message   string
}

func FallbackActive(in *Input) bool {
	return ruleLevel(in, arenaFallback) != Allow
}

func RunFallback(in *Input, found []FallbackFinding) []Result {
	r := arenaFallback
	r.Check = func(pass *Pass) {
		for _, f := range uniqueFallbacks(found) {
			if pos, ok := filePos(in, f.File, f.Line, f.Col); ok {
				pass.Report(Diagnostic{Pos: pos, Message: f.Message})
			}
		}
	}
	return runRule(in, policyOf(in), collectDirectives(in), r)
}

func uniqueFallbacks(found []FallbackFinding) []FallbackFinding {
	sorted := slices.Clone(found)
	slices.SortStableFunc(sorted, func(a, b FallbackFinding) int {
		return cmp.Or(cmp.Compare(a.File, b.File), cmp.Compare(a.Line, b.Line), cmp.Compare(a.Col, b.Col))
	})
	return slices.CompactFunc(sorted, func(a, b FallbackFinding) bool {
		return a.File == b.File && a.Line == b.Line && a.Col == b.Col
	})
}
