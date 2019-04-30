// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"fmt"
	"go/ast"

	"golang.org/x/tools/internal/lsp/snippet"
)

// structFieldSnippets calculates the plain and placeholder snippets for struct literal field names.
func (c *completer) structFieldSnippets(label, detail string) (*snippet.Builder, *snippet.Builder) {
	if !c.inCompositeLiteralField {
		return nil, nil
	}

	lit := c.enclosingCompositeLiteral
	kv := c.enclosingKeyValue

	// If we aren't in a composite literal or are already in a key-value expression,
	// we don't want a snippet.
	if lit == nil || kv != nil {
		return nil, nil
	}
	// First, confirm that we are actually completing a struct field.
	if len(lit.Elts) > 0 {
		i := indexExprAtPos(c.pos, lit.Elts)
		if i >= len(lit.Elts) {
			return nil, nil
		}
		// If the expression is not an identifer, it is not a struct field name.
		if _, ok := lit.Elts[i].(*ast.Ident); !ok {
			return nil, nil
		}
	}

	plain, placeholder := &snippet.Builder{}, &snippet.Builder{}
	label = fmt.Sprintf("%s: ", label)

	// A plain snippet turns "Foo{Ba<>" into "Foo{Bar: <>".
	plain.WriteText(label)
	plain.WritePlaceholder(nil)

	// A placeholder snippet turns "Foo{Ba<>" into "Foo{Bar: <*int*>".
	placeholder.WriteText(label)
	placeholder.WritePlaceholder(func(b *snippet.Builder) {
		b.WriteText(detail)
	})

	// If the cursor position is on a different line from the literal's opening brace,
	// we are in a multiline literal.
	if c.view.FileSet().Position(c.pos).Line != c.view.FileSet().Position(lit.Lbrace).Line {
		plain.WriteText(",")
		placeholder.WriteText(",")
	}

	return plain, placeholder
}

// functionCallSnippets calculates the plain and placeholder snippets for function calls.
func (c *completer) functionCallSnippets(name string, params []string) (*snippet.Builder, *snippet.Builder) {
	for i := 1; i <= 2 && i < len(c.path); i++ {
		call, ok := c.path[i].(*ast.CallExpr)

		// If we are the left side (i.e. "Fun") part of a call expression,
		// we don't want a snippet since there are already parens present.
		if ok && call.Fun == c.path[i-1] {
			return nil, nil
		}
	}

	plain, placeholder := &snippet.Builder{}, &snippet.Builder{}
	label := fmt.Sprintf("%s(", name)

	// A plain snippet turns "someFun<>" into "someFunc(<>)".
	plain.WriteText(label)
	if len(params) > 0 {
		plain.WritePlaceholder(nil)
	}
	plain.WriteText(")")

	// A placeholder snippet turns "someFun<>" into "someFunc(<*i int*>, *s string*)".
	placeholder.WriteText(label)
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
