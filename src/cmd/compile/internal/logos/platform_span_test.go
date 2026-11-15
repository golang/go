// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package logos

import (
	"errors"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// platformSpanSource is the shape of test/logos/run/region_list.lgo that
// broke the eject text: a for-post statement assigning a field read to a
// pointer variable whose declared type forces a MustNonNil check. The
// post statement has no terminator, so a span scanner that eats the loop
// brace wraps the body into the check call.
const platformSpanSource = `package main

type node struct {
	next *node
	v    int
}

func mk[T any]() *T {
	return new(T)
}

func f() {
	head := mk[node]()
	tail := head
	for i := 0; i < 2; i++ {
		n := mk[node]()
		tail.next = n
		tail = n
	}
	tail.next = head
	for h := head.next; h != head; h = h.next {
		_ = h.v
	}
}
`

// TestPlatformSpanEnclosesOperand lowers the for-post shape and asserts
// the lowered text parses and each MustNonNil wrapper encloses exactly
// its recorded operand.
func TestPlatformSpanEnclosesOperand(t *testing.T) {
	path := filepath.Join(t.TempDir(), "platform_span.lgo")
	if err := os.WriteFile(path, []byte(platformSpanSource), 0666); err != nil {
		t.Fatal(err)
	}
	var perrs []error
	f, err := syntax.ParseFile(path, func(err error) { perrs = append(perrs, err) }, nil, syntax.CheckBranches|syntax.LogosMode)
	if err != nil {
		t.Fatal(errors.Join(append(perrs, err)...))
	}
	info := &types2.Info{
		Defs:       make(map[*syntax.Name]types2.Object),
		Uses:       make(map[*syntax.Name]types2.Object),
		Types:      make(map[syntax.Expr]types2.TypeAndValue),
		Selections: make(map[*syntax.SelectorExpr]*types2.Selection),
		Scopes:     make(map[syntax.Node]*types2.Scope),
	}
	var cerrs []error
	conf := types2.Config{
		Error: func(err error) {
			if types2.LogosWarnOnly(err) {
				return
			}
			cerrs = append(cerrs, err)
		},
		Logos: true,
	}
	data := []byte(platformSpanSource)
	files := []*syntax.File{f}
	src := map[*syntax.File][]byte{f: data}
	pkg, _ := conf.Check("main", files, info)
	if err := errors.Join(cerrs...); err != nil {
		t.Fatalf("type check: %v", err)
	}
	_, edits := Lower(pkg, info, files, src)
	out, err := ApplyEdits(data, f.Pos().FileBase(), edits[f])
	if err != nil {
		t.Fatalf("ApplyEdits: %v", err)
	}
	reparseLowered(t, "platform_span.lgo", out)
	text := string(out)
	for _, want := range []string{"MustNonNil(n, &", "MustNonNil(h.next, &"} {
		if !strings.Contains(text, want) {
			t.Errorf("lowered text misses %q:\n%s", want, text)
		}
	}
	for _, bad := range []string{"MustNonNil(n {", "MustNonNil(h.next {"} {
		if strings.Contains(text, bad) {
			t.Errorf("lowered text wraps past its operand %q:\n%s", bad, text)
		}
	}
}
