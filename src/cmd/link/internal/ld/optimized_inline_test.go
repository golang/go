// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ld

import (
	"debug/dwarf"
	"internal/testenv"
	"path/filepath"
	"testing"

	"cmd/link/internal/dwtest"
)

// Keep the line numbers of the issue 82076 reproducer: the initializer call
// is on line 17 and the two inlineThis calls are on lines 18 and 19.
const optimizedInlineSource = `package main

import "fmt"

func inlineThis(a int) int {
	z := a * a
	return f(z + a)
}

func initialize(a, b *int) {
	*a = 3
	*b = 4
}

func main() {
	var a, b int
	initialize(&a, &b)
	a = inlineThis(a)
	b = inlineThis(b)
	fmt.Printf("%d %d\n", a, b)
}

//go:noinline
func f(x int) int {
	return x
}
`

// Check concrete inline instances and their declarations, even when their
// code or values have been optimized away. Value locations are tested separately.
func TestOptimizedInline(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	mustHaveDWARF(t)
	t.Parallel()

	d, ex := gobuildAndExamine(t, optimizedInlineSource, DefaultOpt)
	main := findSubprogramDIE(t, ex, "main.main")
	mainIdx := ex.IdxFromOffset(main.Offset)
	mainRanges, err := d.Ranges(main)
	if err != nil {
		t.Fatal(err)
	}
	lr, err := d.LineReader(ex.ParentCU(mainIdx))
	if err != nil {
		t.Fatal(err)
	}
	if lr == nil {
		t.Fatal("missing line table")
	}

	wantCalls := map[int64]string{17: "main.initialize", 18: "main.inlineThis", 19: "main.inlineThis"}
	for _, call := range ex.Children(mainIdx) {
		if call.Tag != dwarf.TagInlinedSubroutine {
			continue
		}
		line, _ := call.Val(dwarf.AttrCallLine).(int64)
		want, ok := wantCalls[line]
		if !ok {
			continue // e.g., fmt.Printf
		}
		origin := inlineOrigin(t, ex, call)
		if got := origin.Val(dwarf.AttrName); got != want {
			t.Errorf("call on line %d: origin name = %v, want %s", line, got, want)
		}
		delete(wantCalls, line)
		ranges, err := d.Ranges(call)
		if err != nil {
			t.Fatal(err)
		}
		if len(ranges) == 0 {
			t.Errorf("call on line %d has no ranges", line)
		}
		for _, r := range ranges {
			if r[0] >= r[1] || !inlineRangeContained(mainRanges, r) {
				t.Errorf("call on line %d: invalid range %x in main ranges %x", line, r, mainRanges)
			}
		}
		if line == 17 {
			// The otherwise empty initialize call must still provide a
			// breakpoint location, not just an empty inline DIE.
			statement := false
			for _, r := range ranges {
				var row dwarf.LineEntry
				if err := lr.SeekPC(r[0], &row); err != nil {
					t.Fatal(err)
				}
				statement = statement || row.IsStmt && row.Line == 17 &&
					row.File != nil && filepath.Base(row.File.Name) == "test.go"
			}
			if !statement {
				t.Error("initialize has no executable call-site statement")
			}
			continue
		}

		// Each inlineThis instance must describe its own parameter and local,
		// even when constant folding removes all SSA names for them.
		wantVars := map[string]dwarf.Tag{"a": dwarf.TagFormalParameter, "z": dwarf.TagVariable}
		for _, v := range ex.Children(ex.IdxFromOffset(call.Offset)) {
			if v.Tag != dwarf.TagVariable && v.Tag != dwarf.TagFormalParameter {
				continue
			}
			vo := inlineOrigin(t, ex, v)
			name, _ := vo.Val(dwarf.AttrName).(string)
			if tag, ok := wantVars[name]; ok {
				if v.Tag != tag || ex.Parent(ex.IdxFromOffset(vo.Offset)) != origin {
					t.Errorf("call on line %d: incorrect tag or abstract parent for %s", line, name)
				}
				delete(wantVars, name)
			}
		}
		if len(wantVars) != 0 {
			t.Errorf("call on line %d: missing concrete declarations: %v", line, wantVars)
		}
	}
	for line, name := range wantCalls {
		t.Errorf("missing concrete %s inline on line %d", name, line)
	}
}

func inlineOrigin(t *testing.T, ex *dwtest.Examiner, e *dwarf.Entry) *dwarf.Entry {
	t.Helper()
	off, ok := e.Val(dwarf.AttrAbstractOrigin).(dwarf.Offset)
	if !ok {
		t.Fatalf("DIE at %x has no abstract origin", e.Offset)
	}
	o := ex.EntryFromOffset(off)
	if o == nil {
		t.Fatalf("DIE at %x has unresolved abstract origin %x", e.Offset, off)
	}
	return o
}

func inlineRangeContained(ranges [][2]uint64, r [2]uint64) bool {
	for _, outer := range ranges {
		if outer[0] <= r[0] && r[1] <= outer[1] {
			return true
		}
	}
	return false
}
