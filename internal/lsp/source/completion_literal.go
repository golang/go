// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"go/ast"
	"go/token"
	"go/types"
	"strings"
	"unicode"

	"golang.org/x/tools/internal/lsp/diff"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/snippet"
	"golang.org/x/tools/internal/span"
	"golang.org/x/tools/internal/telemetry/log"
)

// literal generates composite literal, function literal, and make()
// completion items.
func (c *completer) literal(literalType types.Type, imp *importInfo) {
	// Don't provide literal candidates for variadic function arguments.
	// For example, don't provide "[]interface{}{}" in "fmt.Print(<>)".
	if c.expectedType.variadic {
		return
	}

	expType := c.expectedType.objType

	// Avoid literal candidates if the expected type is an empty
	// interface. It isn't very useful to suggest a literal candidate of
	// every possible type.
	if expType != nil && isEmptyInterface(expType) {
		return
	}

	// We handle unnamed literal completions explicitly before searching
	// for candidates. Avoid named-type literal completions for
	// unnamed-type expected type since that results in duplicate
	// candidates. For example, in
	//
	// type mySlice []int
	// var []int = <>
	//
	// don't offer "mySlice{}" since we have already added a candidate
	// of "[]int{}".
	if _, named := literalType.(*types.Named); named && expType != nil {
		if _, named := deref(expType).(*types.Named); !named {
			return
		}
	}

	// Check if an object of type literalType or *literalType would
	// match our expected type.
	if !c.matchingType(literalType) {
		literalType = types.NewPointer(literalType)

		if !c.matchingType(literalType) {
			return
		}
	}

	ptr, isPointer := literalType.(*types.Pointer)
	if isPointer {
		literalType = ptr.Elem()
	}

	var (
		qf  = c.qf
		sel = enclosingSelector(c.path, c.pos)
	)

	// Don't qualify the type name if we are in a selector expression
	// since the package name is already present.
	if sel != nil {
		qf = func(_ *types.Package) string { return "" }
	}

	typeName := types.TypeString(literalType, qf)

	// A type name of "[]int" doesn't work very will with the matcher
	// since "[" isn't a valid identifier prefix. Here we strip off the
	// slice (and array) prefix yielding just "int".
	matchName := typeName
	switch t := literalType.(type) {
	case *types.Slice:
		matchName = types.TypeString(t.Elem(), qf)
	case *types.Array:
		matchName = types.TypeString(t.Elem(), qf)
	}

	addlEdits, err := c.importEdits(imp)
	if err != nil {
		log.Error(c.ctx, "error adding import for literal candidate", err)
		return
	}

	// If prefix matches the type name, client may want a composite literal.
	if score := c.matcher.Score(matchName); score >= 0 {
		if isPointer {
			if sel != nil {
				// If we are in a selector we must place the "&" before the selector.
				// For example, "foo.B<>" must complete to "&foo.Bar{}", not
				// "foo.&Bar{}".
				edits, err := referenceEdit(c.view.Session().Cache().FileSet(), c.mapper, sel)
				if err != nil {
					log.Error(c.ctx, "error making edit for literal pointer completion", err)
					return
				}
				addlEdits = append(addlEdits, edits...)
			} else {
				// Otherwise we can stick the "&" directly before the type name.
				typeName = "&" + typeName
			}
		}

		switch t := literalType.Underlying().(type) {
		case *types.Struct, *types.Array, *types.Slice, *types.Map:
			c.compositeLiteral(t, typeName, float64(score), addlEdits)
		case *types.Basic, *types.Signature:
			// Add a literal completion for basic types that implement our
			// expected interface (e.g. named string type http.Dir
			// implements http.FileSystem).
			if isInterface(expType) {
				c.basicLiteral(t, typeName, float64(score), addlEdits)
			}
		}
	}

	// If prefix matches "make", client may want a "make()"
	// invocation. We also include the type name to allow for more
	// flexible fuzzy matching.
	if score := c.matcher.Score("make." + matchName); !isPointer && score >= 0 {
		switch literalType.Underlying().(type) {
		case *types.Slice:
			// The second argument to "make()" for slices is required, so default to "0".
			c.makeCall(typeName, "0", float64(score), addlEdits)
		case *types.Map, *types.Chan:
			// Maps and channels don't require the second argument, so omit
			// to keep things simple for now.
			c.makeCall(typeName, "", float64(score), addlEdits)
		}
	}

	// If prefix matches "func", client may want a function literal.
	if score := c.matcher.Score("func"); !isPointer && score >= 0 && !isInterface(expType) {
		switch t := literalType.Underlying().(type) {
		case *types.Signature:
			c.functionLiteral(t, float64(score))
		}
	}
}

// referenceEdit produces text edits that prepend a "&" operator to the
// specified node.
func referenceEdit(fset *token.FileSet, m *protocol.ColumnMapper, node ast.Node) ([]protocol.TextEdit, error) {
	rng := span.Range{
		FileSet: fset,
		Start:   node.Pos(),
		End:     node.Pos(),
	}
	spn, err := rng.Span()
	if err != nil {
		return nil, err
	}
	return ToProtocolEdits(m, []diff.TextEdit{{
		Span:    spn,
		NewText: "&",
	}})
}

// literalCandidateScore is the base score for literal candidates.
// Literal candidates match the expected type so they should be high
// scoring, but we want them ranked below lexical objects of the
// correct type, so scale down highScore.
const literalCandidateScore = highScore / 2

// functionLiteral adds a function literal completion item for the
// given signature.
func (c *completer) functionLiteral(sig *types.Signature, matchScore float64) {
	snip := &snippet.Builder{}
	snip.WriteText("func(")
	seenParamNames := make(map[string]bool)
	for i := 0; i < sig.Params().Len(); i++ {
		if i > 0 {
			snip.WriteText(", ")
		}

		p := sig.Params().At(i)
		name := p.Name()

		// If the parameter has no name in the signature, we need to try
		// come up with a parameter name.
		if name == "" {
			// Our parameter names are guesses, so they must be placeholders
			// for easy correction. If placeholders are disabled, don't
			// offer the completion.
			if !c.opts.Placeholders {
				return
			}

			// Try abbreviating named types. If the type isn't named, or the
			// abbreviation duplicates a previous name, give up and use
			// "_". The user will have to provide a name for this parameter
			// in order to use it.
			if named, ok := deref(p.Type()).(*types.Named); ok {
				name = abbreviateCamel(named.Obj().Name())
				if seenParamNames[name] {
					name = "_"
				} else {
					seenParamNames[name] = true
				}
			} else {
				name = "_"
			}
			snip.WritePlaceholder(func(b *snippet.Builder) {
				b.WriteText(name)
			})
		} else {
			snip.WriteText(name)
		}

		// If the following param's type is identical to this one, omit
		// this param's type string. For example, emit "i, j int" instead
		// of "i int, j int".
		if i == sig.Params().Len()-1 || !types.Identical(p.Type(), sig.Params().At(i+1).Type()) {
			snip.WriteText(" ")
			typeStr := types.TypeString(p.Type(), c.qf)
			if sig.Variadic() && i == sig.Params().Len()-1 {
				typeStr = strings.Replace(typeStr, "[]", "...", 1)
			}
			snip.WriteText(typeStr)
		}
	}
	snip.WriteText(")")

	results := sig.Results()
	if results.Len() > 0 {
		snip.WriteText(" ")
	}

	resultsNeedParens := results.Len() > 1 ||
		results.Len() == 1 && results.At(0).Name() != ""

	if resultsNeedParens {
		snip.WriteText("(")
	}
	for i := 0; i < results.Len(); i++ {
		if i > 0 {
			snip.WriteText(", ")
		}
		r := results.At(i)
		if name := r.Name(); name != "" {
			snip.WriteText(name + " ")
		}
		snip.WriteText(types.TypeString(r.Type(), c.qf))
	}
	if resultsNeedParens {
		snip.WriteText(")")
	}

	snip.WriteText(" {")
	snip.WriteFinalTabstop()
	snip.WriteText("}")

	c.items = append(c.items, CompletionItem{
		Label:   "func(...) {}",
		Score:   matchScore * literalCandidateScore,
		Kind:    protocol.VariableCompletion,
		snippet: snip,
	})
}

// abbreviateCamel abbreviates camel case identifiers into
// abbreviations. For example, "fooBar" is abbreviated "fb".
func abbreviateCamel(s string) string {
	var (
		b            strings.Builder
		useNextUpper bool
	)
	for i, r := range s {
		if i == 0 {
			b.WriteRune(unicode.ToLower(r))
		}

		if unicode.IsUpper(r) {
			if useNextUpper {
				b.WriteRune(unicode.ToLower(r))
				useNextUpper = false
			}
		} else {
			useNextUpper = true
		}
	}

	return b.String()
}

// compositeLiteral adds a composite literal completion item for the given typeName.
func (c *completer) compositeLiteral(T types.Type, typeName string, matchScore float64, edits []protocol.TextEdit) {
	snip := &snippet.Builder{}
	snip.WriteText(typeName + "{")
	// Don't put the tab stop inside the composite literal curlies "{}"
	// for structs that have no fields.
	if strct, ok := T.(*types.Struct); !ok || strct.NumFields() > 0 {
		snip.WriteFinalTabstop()
	}
	snip.WriteText("}")

	nonSnippet := typeName + "{}"

	c.items = append(c.items, CompletionItem{
		Label:               nonSnippet,
		InsertText:          nonSnippet,
		Score:               matchScore * literalCandidateScore,
		Kind:                protocol.VariableCompletion,
		AdditionalTextEdits: edits,
		snippet:             snip,
	})
}

// basicLiteral adds a literal completion item for the given basic
// type name typeName.
func (c *completer) basicLiteral(T types.Type, typeName string, matchScore float64, edits []protocol.TextEdit) {
	snip := &snippet.Builder{}
	snip.WriteText(typeName + "(")
	snip.WriteFinalTabstop()
	snip.WriteText(")")

	nonSnippet := typeName + "()"

	c.items = append(c.items, CompletionItem{
		Label:               nonSnippet,
		InsertText:          nonSnippet,
		Detail:              T.String(),
		Score:               matchScore * literalCandidateScore,
		Kind:                protocol.VariableCompletion,
		AdditionalTextEdits: edits,
		snippet:             snip,
	})
}

// makeCall adds a completion item for a "make()" call given a specific type.
func (c *completer) makeCall(typeName string, secondArg string, matchScore float64, edits []protocol.TextEdit) {
	// Keep it simple and don't add any placeholders for optional "make()" arguments.

	snip := &snippet.Builder{}
	snip.WriteText("make(" + typeName)
	if secondArg != "" {
		snip.WriteText(", ")
		snip.WritePlaceholder(func(b *snippet.Builder) {
			if c.opts.Placeholders {
				b.WriteText(secondArg)
			}
		})
	}
	snip.WriteText(")")

	var nonSnippet strings.Builder
	nonSnippet.WriteString("make(" + typeName)
	if secondArg != "" {
		nonSnippet.WriteString(", ")
		nonSnippet.WriteString(secondArg)
	}
	nonSnippet.WriteByte(')')

	c.items = append(c.items, CompletionItem{
		Label:               nonSnippet.String(),
		InsertText:          nonSnippet.String(),
		Score:               matchScore * literalCandidateScore,
		Kind:                protocol.FunctionCompletion,
		AdditionalTextEdits: edits,
		snippet:             snip,
	})
}
