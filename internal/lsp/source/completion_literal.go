// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"go/types"
	"strings"

	"golang.org/x/tools/internal/lsp/snippet"
)

// literal generates composite literal and make() completion items.
func (c *completer) literal(literalType types.Type) {
	if c.expectedType.objType == nil {
		return
	}

	// Don't provide literal candidates for variadic function arguments.
	// For example, don't provide "[]interface{}{}" in "fmt.Print(<>)".
	if c.expectedType.variadic {
		return
	}

	// Avoid literal candidates if the expected type is an empty
	// interface. It isn't very useful to suggest a literal candidate of
	// every possible type.
	if isEmptyInterface(c.expectedType.objType) {
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
	if _, named := literalType.(*types.Named); named {
		if _, named := deref(c.expectedType.objType).(*types.Named); !named {
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

	typeName := types.TypeString(literalType, c.qf)

	// A type name of "[]int" doesn't work very will with the matcher
	// since "[" isn't a valid identifier prefix. Here we strip off the
	// slice (and array) prefix yielding just "int".
	matchName := typeName
	switch t := literalType.(type) {
	case *types.Slice:
		matchName = types.TypeString(t.Elem(), c.qf)
	case *types.Array:
		matchName = types.TypeString(t.Elem(), c.qf)
	}

	// If prefix matches the type name, client may want a composite literal.
	if score := c.matcher.Score(matchName); score >= 0 {
		if isPointer {
			typeName = "&" + typeName
		}

		switch t := literalType.Underlying().(type) {
		case *types.Struct, *types.Array, *types.Slice, *types.Map:
			c.compositeLiteral(t, typeName, float64(score))
		}
	}

	// If prefix matches "make", client may want a "make()"
	// invocation. We also include the type name to allow for more
	// flexible fuzzy matching.
	if score := c.matcher.Score("make." + matchName); !isPointer && score >= 0 {
		switch literalType.Underlying().(type) {
		case *types.Slice:
			// The second argument to "make()" for slices is required, so default to "0".
			c.makeCall(typeName, "0", float64(score))
		case *types.Map, *types.Chan:
			// Maps and channels don't require the second argument, so omit
			// to keep things simple for now.
			c.makeCall(typeName, "", float64(score))
		}
	}
}

// literalCandidateScore is the base score for literal candidates.
// Literal candidates match the expected type so they should be high
// scoring, but we want them ranked below lexical objects of the
// correct type, so scale down highScore.
const literalCandidateScore = highScore / 2

// compositeLiteral adds a composite literal completion item for the given typeName.
func (c *completer) compositeLiteral(T types.Type, typeName string, matchScore float64) {
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
		Label:      nonSnippet,
		InsertText: nonSnippet,
		Score:      matchScore * literalCandidateScore,
		Kind:       VariableCompletionItem,
		snippet:    snip,
	})
}

// makeCall adds a completion item for a "make()" call given a specific type.
func (c *completer) makeCall(typeName string, secondArg string, matchScore float64) {
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
		Label:      nonSnippet.String(),
		InsertText: nonSnippet.String(),
		Score:      matchScore * literalCandidateScore,
		Kind:       FunctionCompletionItem,
		snippet:    snip,
	})
}
