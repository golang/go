// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package completion

import (
	"go/ast"
	"go/token"
	"go/types"
	"strings"
	"unicode"
	"unicode/utf8"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/snippet"
	"golang.org/x/tools/gopls/internal/lsp/source"
	"golang.org/x/tools/gopls/internal/span"
)

// some definitions can be completed
// So far, TestFoo(t *testing.T), TestMain(m *testing.M)
// BenchmarkFoo(b *testing.B), FuzzFoo(f *testing.F)

// path[0] is known to be *ast.Ident
func definition(path []ast.Node, obj types.Object, tokFile *token.File, fh source.FileHandle) ([]CompletionItem, *Selection) {
	if _, ok := obj.(*types.Func); !ok {
		return nil, nil // not a function at all
	}
	if !strings.HasSuffix(fh.URI().Filename(), "_test.go") {
		return nil, nil
	}

	name := path[0].(*ast.Ident).Name
	if len(name) == 0 {
		// can't happen
		return nil, nil
	}
	pos := path[0].Pos()
	sel := &Selection{
		content: "",
		cursor:  pos,
		rng:     span.NewRange(tokFile, pos, pos),
	}
	var ans []CompletionItem

	// Always suggest TestMain, if possible
	if strings.HasPrefix("TestMain", name) {
		ans = []CompletionItem{defItem("TestMain(m *testing.M)", obj)}
	}

	// If a snippet is possible, suggest it
	if strings.HasPrefix("Test", name) {
		ans = append(ans, defSnippet("Test", "Xxx", "(t *testing.T)", obj))
		return ans, sel
	} else if strings.HasPrefix("Benchmark", name) {
		ans = append(ans, defSnippet("Benchmark", "Xxx", "(b *testing.B)", obj))
		return ans, sel
	} else if strings.HasPrefix("Fuzz", name) {
		ans = append(ans, defSnippet("Fuzz", "Xxx", "(f *testing.F)", obj))
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

func defMatches(name, pat string, path []ast.Node, arg string) string {
	idx := strings.Index(name, pat)
	if idx < 0 {
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
	if fp != nil && len(fp.List) > 0 {
		// signature already there, minimal suggestion
		return name
	}
	// suggesting signature too
	return name + arg
}

func defSnippet(prefix, placeholder, suffix string, obj types.Object) CompletionItem {
	var sn snippet.Builder
	sn.WriteText(prefix)
	if placeholder != "" {
		sn.WritePlaceholder(func(b *snippet.Builder) { b.WriteText(placeholder) })
	}
	sn.WriteText(suffix + " {\n")
	sn.WriteFinalTabstop()
	sn.WriteText("\n}")
	return CompletionItem{
		Label:         prefix + placeholder + suffix,
		Detail:        "tab, type the rest of the name, then tab",
		Kind:          protocol.FunctionCompletion,
		Depth:         0,
		Score:         10,
		snippet:       &sn,
		Documentation: prefix + " test function",
		obj:           obj,
	}
}
func defItem(val string, obj types.Object) CompletionItem {
	return CompletionItem{
		Label:         val,
		InsertText:    val,
		Kind:          protocol.FunctionCompletion,
		Depth:         0,
		Score:         9, // prefer the snippets when available
		Documentation: "complete the parameter",
		obj:           obj,
	}
}
