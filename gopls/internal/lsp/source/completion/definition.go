// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package completion

import (
	"go/ast"
	"go/types"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/snippet"
	"golang.org/x/tools/gopls/internal/lsp/source"
)

// some function definitions in test files can be completed
// So far, TestFoo(t *testing.T), TestMain(m *testing.M)
// BenchmarkFoo(b *testing.B), FuzzFoo(f *testing.F)

// path[0] is known to be *ast.Ident
func definition(path []ast.Node, obj types.Object, pgf *source.ParsedGoFile) ([]CompletionItem, *Selection) {
	if _, ok := obj.(*types.Func); !ok {
		return nil, nil // not a function at all
	}
	if !strings.HasSuffix(pgf.URI.Filename(), "_test.go") {
		return nil, nil // not a test file
	}

	name := path[0].(*ast.Ident).Name
	if len(name) == 0 {
		// can't happen
		return nil, nil
	}
	start := path[0].Pos()
	end := path[0].End()
	sel := &Selection{
		content: "",
		cursor:  start,
		tokFile: pgf.Tok,
		start:   start,
		end:     end,
		mapper:  pgf.Mapper,
	}
	var ans []CompletionItem
	var hasParens bool
	n, ok := path[1].(*ast.FuncDecl)
	if !ok {
		return nil, nil // can't happen
	}
	if n.Recv != nil {
		return nil, nil // a method, not a function
	}
	t := n.Type.Params
	if t.Closing != t.Opening {
		hasParens = true
	}

	// Always suggest TestMain, if possible
	if strings.HasPrefix("TestMain", name) {
		if hasParens {
			ans = append(ans, defItem("TestMain", obj))
		} else {
			ans = append(ans, defItem("TestMain(m *testing.M)", obj))
		}
	}

	// If a snippet is possible, suggest it
	if strings.HasPrefix("Test", name) {
		if hasParens {
			ans = append(ans, defItem("Test", obj))
		} else {
			ans = append(ans, defSnippet("Test", "(t *testing.T)", obj))
		}
		return ans, sel
	} else if strings.HasPrefix("Benchmark", name) {
		if hasParens {
			ans = append(ans, defItem("Benchmark", obj))
		} else {
			ans = append(ans, defSnippet("Benchmark", "(b *testing.B)", obj))
		}
		return ans, sel
	} else if strings.HasPrefix("Fuzz", name) {
		if hasParens {
			ans = append(ans, defItem("Fuzz", obj))
		} else {
			ans = append(ans, defSnippet("Fuzz", "(f *testing.F)", obj))
		}
		return ans, sel
	}

	// Fill in the argument for what the user has already typed
	if got := defMatches(name, "Test", path, "(t *testing.T)"); got != "" {
		ans = append(ans, defItem(got, obj))
	} else if got := defMatches(name, "Benchmark", path, "(b *testing.B)"); got != "" {
		ans = append(ans, defItem(got, obj))
	} else if got := defMatches(name, "Fuzz", path, "(f *testing.F)"); got != "" {
		ans = append(ans, defItem(got, obj))
	}
	return ans, sel
}

// defMatches returns text for defItem, never for defSnippet
func defMatches(name, pat string, path []ast.Node, arg string) string {
	if !strings.HasPrefix(name, pat) {
		return ""
	}
	c, _ := utf8.DecodeRuneInString(name[len(pat):])
	if unicode.IsLower(c) {
		return ""
	}
	fd, ok := path[1].(*ast.FuncDecl)
	if !ok {
		// we don't know what's going on
		return ""
	}
	fp := fd.Type.Params
	if len(fp.List) > 0 {
		// signature already there, nothing to suggest
		return ""
	}
	if fp.Opening != fp.Closing {
		// nothing: completion works on words, not easy to insert arg
		return ""
	}
	// suggesting signature too
	return name + arg
}

func defSnippet(prefix, suffix string, obj types.Object) CompletionItem {
	var sn snippet.Builder
	sn.WriteText(prefix)
	sn.WritePlaceholder(func(b *snippet.Builder) { b.WriteText("Xxx") })
	sn.WriteText(suffix + " {\n\t")
	sn.WriteFinalTabstop()
	sn.WriteText("\n}")
	return CompletionItem{
		Label:         prefix + "Xxx" + suffix,
		Detail:        "tab, type the rest of the name, then tab",
		Kind:          protocol.FunctionCompletion,
		Depth:         0,
		Score:         10,
		snippet:       &sn,
		Documentation: prefix + " test function",
		isSlice:       isSlice(obj),
	}
}
func defItem(val string, obj types.Object) CompletionItem {
	return CompletionItem{
		Label:         val,
		InsertText:    val,
		Kind:          protocol.FunctionCompletion,
		Depth:         0,
		Score:         9, // prefer the snippets when available
		Documentation: "complete the function name",
		isSlice:       isSlice(obj),
	}
}
