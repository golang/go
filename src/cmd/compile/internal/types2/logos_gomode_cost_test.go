// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2_test

import (
	"cmd/compile/internal/syntax"
	. "cmd/compile/internal/types2"
	"fmt"
	"strings"
	"testing"
)

func methodCallSource(recv string, calls int) string {
	var b strings.Builder
	b.WriteString("package p\n\ntype T struct {\n\ta [10]int\n\tn int\n}\n\n")
	fmt.Fprintf(&b, "func (t %s) M(i int) int {\n\ts := 0\n", recv)
	for k := range 200 {
		fmt.Fprintf(&b, "\ts += t.a[(i+%d)%%10] + t.n\n", k)
	}
	b.WriteString("\treturn s\n}\n\nfunc f(t *T) int {\n\ts := 0\n")
	for k := range calls {
		fmt.Fprintf(&b, "\ts += t.M(%d)\n", k)
	}
	b.WriteString("\treturn s\n}\n")
	return b.String()
}

func checkAllocs(t *testing.T, src string) float64 {
	t.Helper()
	file, err := syntax.Parse(syntax.NewFileBase("p.go"), strings.NewReader(src), nil, nil, 0)
	if err != nil {
		t.Fatal(err)
	}
	var failed error
	conf := Config{Error: func(err error) { failed = err }}
	allocs := testing.AllocsPerRun(3, func() {
		info := &Info{Uses: make(map[*syntax.Name]Object), Types: make(map[syntax.Expr]TypeAndValue)}
		if _, err := conf.Check("p", []*syntax.File{file}, info); err != nil {
			failed = err
		}
	})
	if failed != nil {
		t.Fatal(failed)
	}
	return allocs
}

func TestGoModeCallCostDoesNotGrowWithCalleeBody(t *testing.T) {
	const calls = 600
	pointer := checkAllocs(t, methodCallSource("*T", calls))
	value := checkAllocs(t, methodCallSource("T", calls))
	if pointer > value*3/2 {
		t.Errorf("checking %d pointer-receiver calls allocates %.0f, value-receiver calls %.0f", calls, pointer, value)
	}
}
