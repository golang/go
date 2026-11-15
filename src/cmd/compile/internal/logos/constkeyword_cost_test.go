// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"fmt"
	"runtime"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
)

func constKeywordMallocs(t *testing.T, n int) uint64 {
	t.Helper()
	var b strings.Builder
	b.WriteString("package main\n\n")
	for i := range n {
		fmt.Fprintf(&b, "const c%d = []int{%d, %d}\n", i, i, i+1)
	}
	b.WriteString("\nfunc main() {}\n")
	src := []byte(b.String())
	base := syntax.NewFileBase("cost.lgo")
	file := ejectParse(t, base, src, syntax.LogosMode|syntax.CheckBranches)
	pkg, info := ejectCheck(t, file)
	var before, after runtime.MemStats
	runtime.ReadMemStats(&before)
	ejectLower(t, pkg, info, file, src)
	runtime.ReadMemStats(&after)
	return after.Mallocs - before.Mallocs
}

func TestConstKeywordCostIsLinear(t *testing.T) {
	small := constKeywordMallocs(t, 1000)
	large := constKeywordMallocs(t, 4000)
	if large > 6*small {
		t.Fatalf("lowering 4000 const declarations made %d allocations and 1000 made %d: more than 6 times as many", large, small)
	}
}
