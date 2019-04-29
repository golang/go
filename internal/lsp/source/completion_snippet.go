// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"go/ast"

	"golang.org/x/tools/internal/lsp/snippet"
)

// structField calculates the plain and placeholder snippets for struct literal
// field names as in "Foo{Ba<>".
func (c *completer) structFieldSnippet(label, detail string) (*snippet.Builder, *snippet.Builder) {
	if !c.inCompositeLiteralField {
		return nil, nil
	}

	cl := c.enclosingCompositeLiteral
	kv := c.enclosingKeyValue

	// If we aren't in a composite literal or are already in a key/value
	// expression, we don't want a snippet.
	if cl == nil || kv != nil {
		return nil, nil
	}

	if len(cl.Elts) > 0 {
		i := indexExprAtPos(c.pos, cl.Elts)
		if i >= len(cl.Elts) {
			return nil, nil
		}

		// If our expression is not an identifer, we know it isn't a
		// struct field name.
		if _, ok := cl.Elts[i].(*ast.Ident); !ok {
			return nil, nil
		}
	}

	// It is a multi-line literal if pos is not on the same line as the literal's
	// opening brace.
	multiLine := c.fset.Position(c.pos).Line != c.fset.Position(cl.Lbrace).Line

	// Plain snippet will turn "Foo{Ba<>" into "Foo{Bar: <>"
	plain := &snippet.Builder{}
	plain.WriteText(label + ": ")
	plain.WritePlaceholder(nil)
	if multiLine {
		plain.WriteText(",")
	}

	// Placeholder snippet will turn "Foo{Ba<>" into "Foo{Bar: *int*"
	placeholder := &snippet.Builder{}
	placeholder.WriteText(label + ": ")
	placeholder.WritePlaceholder(func(b *snippet.Builder) {
		b.WriteText(detail)
	})
	if multiLine {
		placeholder.WriteText(",")
	}

	return plain, placeholder
}

// funcCall calculates the plain and placeholder snippets for function calls.
func (c *completer) funcCallSnippet(funcName string, params []string) (*snippet.Builder, *snippet.Builder) {
	for i := 1; i <= 2 && i < len(c.path); i++ {
		call, ok := c.path[i].(*ast.CallExpr)
		// If we are the left side (i.e. "Fun") part of a call expression,
		// we don't want a snippet since there are already parens present.
		if ok && call.Fun == c.path[i-1] {
			return nil, nil
		}
	}

	// Plain snippet turns "someFun<>" into "someFunc(<>)"
	plain := &snippet.Builder{}
	plain.WriteText(funcName + "(")
	if len(params) > 0 {
		plain.WritePlaceholder(nil)
	}
	plain.WriteText(")")

	// Placeholder snippet turns "someFun<>" into "someFunc(*i int*, s string)"
	placeholder := &snippet.Builder{}
	placeholder.WriteText(funcName + "(")
	for i, p := range params {
		if i > 0 {
			placeholder.WriteText(", ")
		}
		placeholder.WritePlaceholder(func(b *snippet.Builder) {
			b.WriteText(p)
		})
	}
	placeholder.WriteText(")")

	return plain, placeholder
}
