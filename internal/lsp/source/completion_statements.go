// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/snippet"
)

// addStatementCandidates adds full statement completion candidates
// appropriate for the current context.
func (c *completer) addStatementCandidates() {
	c.addErrCheckAndReturn()
}

// addErrCheckAndReturn offers a completion candidate of the form:
//
//     if err != nil {
//       return nil, err
//     }
//
// The position must be in a function that returns an error, and the
// statement preceding the position must be an assignment where the
// final LHS object is an error. addErrCheckAndReturn will synthesize
// zero values as necessary to make the return statement valid.
func (c *completer) addErrCheckAndReturn() {
	if len(c.path) < 2 || c.enclosingFunc == nil || !c.opts.placeholders {
		return
	}

	var (
		errorType = types.Universe.Lookup("error").Type()
		result    = c.enclosingFunc.sig.Results()
	)
	// Make sure our enclosing function returns an error.
	if result.Len() == 0 || !types.Identical(result.At(result.Len()-1).Type(), errorType) {
		return
	}

	prevLine := prevStmt(c.pos, c.path)
	if prevLine == nil {
		return
	}

	// Make sure our preceding statement was as assignment.
	assign, _ := prevLine.(*ast.AssignStmt)
	if assign == nil || len(assign.Lhs) == 0 {
		return
	}

	lastAssignee := assign.Lhs[len(assign.Lhs)-1]

	// Make sure the final assignee is an error.
	if !types.Identical(c.pkg.GetTypesInfo().TypeOf(lastAssignee), errorType) {
		return
	}

	var (
		// errText is e.g. "err" in "foo, err := bar()".
		errText = formatNode(c.snapshot.View().Session().Cache().FileSet(), lastAssignee)

		// Whether we need to include the "if" keyword in our candidate.
		needsIf = true
	)

	// "_" isn't a real object.
	if errText == "_" {
		return
	}

	// Below we try to detect if the user has already started typing "if
	// err" so we can replace what they've typed with our complete
	// statement.
	switch n := c.path[0].(type) {
	case *ast.Ident:
		switch c.path[1].(type) {
		case *ast.ExprStmt:
			// This handles:
			//
			//     f, err := os.Open("foo")
			//     i<>

			// Make sure they are typing "if".
			if c.matcher.Score("if") <= 0 {
				return
			}
		case *ast.IfStmt:
			// This handles:
			//
			//     f, err := os.Open("foo")
			//     if er<>

			// Make sure they are typing the error's name.
			if c.matcher.Score(errText) <= 0 {
				return
			}

			needsIf = false
		default:
			return
		}
	case *ast.IfStmt:
		// This handles:
		//
		//     f, err := os.Open("foo")
		//     if <>

		// Avoid false positives by ensuring the if's cond is a bad
		// expression. For example, don't offer the completion in cases
		// like "if <> somethingElse".
		if _, bad := n.Cond.(*ast.BadExpr); !bad {
			return
		}

		// If "if" is our direct prefix, we need to include it in our
		// candidate since the existing "if" will be overwritten.
		needsIf = c.pos == n.Pos()+token.Pos(len("if"))
	}

	// Build up a snippet that looks like:
	//
	//     if err != nil {
	//       return <zero value>, ..., ${1:err}
	//     }
	//
	// We make the error a placeholder so it is easy to alter the error.
	var snip snippet.Builder
	if needsIf {
		snip.WriteText("if ")
	}
	snip.WriteText(fmt.Sprintf("%s != nil {\n\treturn ", errText))

	for i := 0; i < result.Len()-1; i++ {
		snip.WriteText(formatZeroValue(result.At(i).Type(), c.qf))
		snip.WriteText(", ")
	}

	snip.WritePlaceholder(func(b *snippet.Builder) {
		b.WriteText(errText)
	})

	snip.WriteText("\n}")

	label := fmt.Sprintf("%[1]s != nil { return %[1]s }", errText)
	if needsIf {
		label = "if " + label
	}

	c.items = append(c.items, CompletionItem{
		Label: label,
		// There doesn't seem to be a more appropriate kind.
		Kind:    protocol.KeywordCompletion,
		Score:   highScore,
		snippet: &snip,
	})
}
