// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package completion

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/lsp/snippet"
	"golang.org/x/tools/internal/lsp/source"
)

// addStatementCandidates adds full statement completion candidates
// appropriate for the current context.
func (c *completer) addStatementCandidates() {
	c.addErrCheckAndReturn()
	c.addAssignAppend()
}

// addAssignAppend offers a completion candidate of the form:
//
//     someSlice = append(someSlice, )
//
// It will offer the "append" completion in two situations:
//
// 1. Position is in RHS of assign, prefix matches "append", and
//    corresponding LHS object is a slice. For example,
//    "foo = ap<>" completes to "foo = append(foo, )".
//
// Or
//
// 2. Prefix is an ident or selector in an *ast.ExprStmt (i.e.
//    beginning of statement), and our best matching candidate is a
//    slice. For example: "foo.ba" completes to "foo.bar = append(foo.bar, )".
func (c *completer) addAssignAppend() {
	if len(c.path) < 3 {
		return
	}

	ident, _ := c.path[0].(*ast.Ident)
	if ident == nil {
		return
	}

	var (
		// sliceText is the full name of our slice object, e.g. "s.abc" in
		// "s.abc = app<>".
		sliceText string
		// needsLHS is true if we need to prepend the LHS slice name and
		// "=" to our candidate.
		needsLHS = false
		fset     = c.snapshot.FileSet()
	)

	switch n := c.path[1].(type) {
	case *ast.AssignStmt:
		// We are already in an assignment. Make sure our prefix matches "append".
		if c.matcher.Score("append") <= 0 {
			return
		}

		exprIdx := exprAtPos(c.pos, n.Rhs)
		if exprIdx == len(n.Rhs) || exprIdx > len(n.Lhs)-1 {
			return
		}

		lhsType := c.pkg.GetTypesInfo().TypeOf(n.Lhs[exprIdx])
		if lhsType == nil {
			return
		}

		// Make sure our corresponding LHS object is a slice.
		if _, isSlice := lhsType.Underlying().(*types.Slice); !isSlice {
			return
		}

		// The name or our slice is whatever's in the LHS expression.
		sliceText = source.FormatNode(fset, n.Lhs[exprIdx])
	case *ast.SelectorExpr:
		// Make sure we are a selector at the beginning of a statement.
		if _, parentIsExprtStmt := c.path[2].(*ast.ExprStmt); !parentIsExprtStmt {
			return
		}

		// So far we only know the first part of our slice name. For
		// example in "s.a<>" we only know our slice begins with "s."
		// since the user could still be typing.
		sliceText = source.FormatNode(fset, n.X) + "."
		needsLHS = true
	case *ast.ExprStmt:
		needsLHS = true
	default:
		return
	}

	var (
		label string
		snip  snippet.Builder
		score = highScore
	)

	if needsLHS {
		// Offer the long form assign + append candidate if our best
		// candidate is a slice.
		bestItem := c.topCandidate()
		if bestItem == nil || bestItem.obj == nil || bestItem.obj.Type() == nil {
			return
		}

		if _, isSlice := bestItem.obj.Type().Underlying().(*types.Slice); !isSlice {
			return
		}

		// Don't rank the full form assign + append candidate above the
		// slice itself.
		score = bestItem.Score - 0.01

		// Fill in rest of sliceText now that we have the object name.
		sliceText += bestItem.Label

		// Fill in the candidate's LHS bits.
		label = fmt.Sprintf("%s = ", bestItem.Label)
		snip.WriteText(label)
	}

	snip.WriteText(fmt.Sprintf("append(%s, ", sliceText))
	snip.WritePlaceholder(nil)
	snip.WriteText(")")

	c.items = append(c.items, CompletionItem{
		Label:   label + fmt.Sprintf("append(%s, )", sliceText),
		Kind:    protocol.FunctionCompletion,
		Score:   score,
		snippet: &snip,
	})
}

// topCandidate returns the strictly highest scoring candidate
// collected so far. If the top two candidates have the same score,
// nil is returned.
func (c *completer) topCandidate() *CompletionItem {
	var bestItem, secondBestItem *CompletionItem
	for i := range c.items {
		if bestItem == nil || c.items[i].Score > bestItem.Score {
			bestItem = &c.items[i]
		} else if secondBestItem == nil || c.items[i].Score > secondBestItem.Score {
			secondBestItem = &c.items[i]
		}
	}

	// If secondBestItem has the same score, bestItem isn't
	// the strict best.
	if secondBestItem != nil && secondBestItem.Score == bestItem.Score {
		return nil
	}

	return bestItem
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
		errText = source.FormatNode(c.snapshot.FileSet(), lastAssignee)

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
