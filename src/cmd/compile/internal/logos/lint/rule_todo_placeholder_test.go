// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package lint

import (
	"internal/testenv"
	"slices"
	"testing"
)

// TestCommentMarker covers which comment texts the rule takes for a marker.
func TestCommentMarker(t *testing.T) {
	for text, want := range map[string]string{
		"// TODO":                  "TODO",
		"//TODO":                   "TODO",
		"//\tTODO: tab":            "TODO",
		"//   FIXME(bob): spaces":  "FIXME",
		"// TODO(alice)":           "TODO",
		"// TODO-ish":              "TODO",
		"// TODO.":                 "TODO",
		"/* TODO: block */":        "TODO",
		"/*FIXME*/":                "FIXME",
		"/*\n\tTODO: later\n*/":    "TODO",
		"// TODOS stay prose":      "",
		"// TODOLIST":              "",
		"// TODO_x":                "",
		"// TODOé":                 "",
		"// FIXMEs":                "",
		"// todo lowercase":        "",
		"// Todo capitalized":      "",
		"// see TODO list":         "",
		"// ":                      "",
		"//":                       "",
		"/**/":                     "",
		"unexpected token":         "",
		"":                         "",
		"/ TODO":                   "",
		"// TODO: a // FIXME: two": "TODO",
	} {
		if got := todoMarker(text); got != want {
			t.Errorf("todoMarker(%q) = %q, want %q", text, got, want)
		}
	}
}

// TestNamesUnfinishedWork covers the text a panic stub must hold.
func TestNamesUnfinishedWork(t *testing.T) {
	for msg, want := range map[string]bool{
		"unimplemented":                  true,
		"Unimplemented":                  true,
		"not implemented":                true,
		"Not Implemented yet":            true,
		"NOT IMPLEMENTED":                true,
		"feature unimplemented here":     true,
		"TODO":                           true,
		"TODO: later":                    true,
		"see TODO":                       true,
		"todo":                           false,
		"implemented":                    false,
		"not yet implemented":            false,
		"unreachable":                    false,
		"":                               false,
		"not  implemented (two spaces)":  false,
		"notimplemented":                 false,
		"implementation is unavailable":  false,
		"Todo":                           false,
		"TODOs":                          true,
		"not implemented\nand more text": true,
	} {
		if got := todoNamesUnfinished(msg); got != want {
			t.Errorf("todoNamesUnfinished(%q) = %v, want %v", msg, got, want)
		}
	}
}

const todoSrc = `package p

import "errors"

// TODO: leading
func a() {}

func b() {
	x := 1 // FIXME: trailing
	_ = x
	/* TODO: block
	   over two lines */
}

func c() {
	panic("unimplemented")
	panic(errors.New("unimplemented"))
	panic("fine")
}

var s = "// TODO: string"
`

// TestTodoPlaceholder checks the findings and the spans of both forms.
func TestTodoPlaceholder(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	for _, kind := range []Kind{KindGo, KindLgo} {
		name := "x.go"
		if kind == KindLgo {
			name = "x.lgo"
		}
		rs := Run(check(t, kind, name, todoSrc))
		want := []string{
			"5 slop::todo_placeholder deny",
			"9 slop::todo_placeholder deny",
			"11 slop::todo_placeholder deny",
			"16 slop::todo_placeholder deny",
		}
		if got := summarize(rs); !slices.Equal(got, want) {
			t.Errorf("%s results = %q, want %q", name, got, want)
			continue
		}
		for i, want := range []string{
			"// TODO: leading",
			"// FIXME: trailing",
			"/* TODO: block\n\t   over two lines */",
			`panic("unimplemented")`,
		} {
			if got := spanText(todoSrc, rs[i]); got != want {
				t.Errorf("%s span %d = %q, want %q", name, i, got, want)
			}
		}
		if got, want := rs[0].Message, "comment starts with TODO: finish the work or remove the marker"; got != want {
			t.Errorf("%s comment message = %q, want %q", name, got, want)
		}
		if got, want := rs[3].Message, `panic stub "unimplemented": implement it or return an error`; got != want {
			t.Errorf("%s panic message = %q, want %q", name, got, want)
		}
		if got := rs[1].Provenance; got.OriginExpr != "// FIXME: trailing" || got.OriginLine != 9 {
			t.Errorf("%s comment provenance = %+v", name, got)
		}
	}
}

// TestTodoPlaceholderTestFiles checks that a test source is skipped, in both
// spellings and both kinds, while the same source in another file reports.
func TestTodoPlaceholderTestFiles(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	for _, tc := range []struct {
		kind Kind
		name string
		want int
	}{
		{KindGo, "x_test.go", 0},
		{KindLgo, "x_test.lgo", 0},
		{KindGo, "x.go", 4},
		{KindLgo, "x.lgo", 4},
		{KindGo, "x_testing.go", 4},
	} {
		got := 0
		for _, r := range Run(check(t, tc.kind, tc.name, todoSrc)) {
			if r.Rule == "slop::todo_placeholder" {
				got++
			}
		}
		if got != tc.want {
			t.Errorf("%s: %d findings, want %d", tc.name, got, tc.want)
		}
	}
}

// TestTodoPlaceholderNoSource checks that a file without bytes keeps its
// panic findings and reports no comment.
func TestTodoPlaceholderNoSource(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	in := check(t, KindGo, "x.go", todoSrc)
	in.Sources = nil
	if got, want := summarize(Run(in)), []string{"16 slop::todo_placeholder deny"}; !slices.Equal(got, want) {
		t.Errorf("results = %q, want %q", got, want)
	}
}

// TestTodoPlaceholderLgoScan checks that a loGo/s file is scanned with its
// own tokens: a quoted string inside an interpolation operand holds a //
// that is data, not a comment, and a Go file with the same text has none.
func TestTodoPlaceholderLgoScan(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	const src = `package p

var s = $"{"// TODO: operand"}"

// TODO: real
func f() {}
`
	got := summarize(Run(check(t, KindLgo, "x.lgo", src)))
	if want := []string{"5 slop::todo_placeholder deny"}; !slices.Equal(got, want) {
		t.Errorf("results = %q, want %q", got, want)
	}
}

// TestTodoPlaceholderAllow checks the //logos:allow scopes for comments: the
// statement or declaration the directive precedes holds the comments in it,
// and a comment between the directive and that node.
func TestTodoPlaceholderAllow(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	const src = `package p

//logos:allow(slop::todo_placeholder)
func a() {
	// TODO: inside
	panic("unimplemented")
}

func b() {
	//logos:allow(slop::todo_placeholder)
	panic("not implemented")
	// TODO: after the covered statement
}

//logos:allow(slop::todo_placeholder)
// TODO: between the directive and the declaration
func c() {}

//logos:allow(slop::todo_placeholder)

// TODO: after a blank line
func d() {}

//logos:allow(slop::discarded_error)
// TODO: another rule
func e() {}
`
	for _, kind := range []Kind{KindGo, KindLgo} {
		name := "x.go"
		if kind == KindLgo {
			name = "x.lgo"
		}
		want := []string{
			"12 slop::todo_placeholder deny",
			"21 slop::todo_placeholder deny",
			"25 slop::todo_placeholder deny",
		}
		if got := summarize(Run(check(t, kind, name, src))); !slices.Equal(got, want) {
			t.Errorf("%s results = %q, want %q", name, got, want)
		}
	}
}

// TestTodoPlaceholderShadowedPanic checks that only the predeclared panic
// counts.
func TestTodoPlaceholderShadowedPanic(t *testing.T) {
	testenv.MustHaveGoBuild(t)
	const src = `package p

func panic(string) {}

func f() {
	panic("unimplemented")
}

func g() {
	panic := func(string) {}
	panic("unimplemented")
}
`
	if got := summarize(Run(check(t, KindGo, "x.go", src))); len(got) != 0 {
		t.Errorf("results = %q, want none", got)
	}
}
