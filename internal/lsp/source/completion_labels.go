// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"go/ast"
	"go/token"
	"math"
)

type labelType int

const (
	labelNone labelType = iota
	labelBreak
	labelContinue
	labelGoto
)

// wantLabelCompletion returns true if we want (only) label
// completions at the position.
func (c *completer) wantLabelCompletion() labelType {
	if _, ok := c.path[0].(*ast.Ident); ok && len(c.path) > 1 {
		// We want a label if we are an *ast.Ident child of a statement
		// that accepts a label, e.g. "break Lo<>".
		return takesLabel(c.path[1])
	}

	return labelNone
}

// takesLabel returns the corresponding labelType if n is a statement
// that accepts a label, otherwise labelNone.
func takesLabel(n ast.Node) labelType {
	if bs, ok := n.(*ast.BranchStmt); ok {
		switch bs.Tok {
		case token.BREAK:
			return labelBreak
		case token.CONTINUE:
			return labelContinue
		case token.GOTO:
			return labelGoto
		}
	}
	return labelNone
}

// labels adds completion items for labels defined in the enclosing
// function.
func (c *completer) labels(lt labelType) {
	if c.enclosingFunc == nil {
		return
	}

	addLabel := func(score float64, l *ast.LabeledStmt) {
		labelObj := c.pkg.GetTypesInfo().ObjectOf(l.Label)
		if labelObj != nil {
			c.found(candidate{obj: labelObj, score: score})
		}
	}

	switch lt {
	case labelBreak, labelContinue:
		// "break" and "continue" only accept labels from enclosing statements.

		for i, p := range c.path {
			switch p := p.(type) {
			case *ast.FuncLit:
				// Labels are function scoped, so don't continue out of functions.
				return
			case *ast.LabeledStmt:
				switch p.Stmt.(type) {
				case *ast.ForStmt, *ast.RangeStmt:
					// Loop labels can be used for "break" or "continue".
					addLabel(highScore*math.Pow(.99, float64(i)), p)
				case *ast.SwitchStmt, *ast.SelectStmt, *ast.TypeSwitchStmt:
					// Switch and select labels can be used only for "break".
					if lt == labelBreak {
						addLabel(highScore*math.Pow(.99, float64(i)), p)
					}
				}
			}
		}
	case labelGoto:
		// Goto accepts any label in the same function not in a nested
		// block. It also doesn't take labels that would jump across
		// variable definitions, but ignore that case for now.
		ast.Inspect(c.enclosingFunc.body, func(n ast.Node) bool {
			if n == nil {
				return false
			}

			switch n := n.(type) {
			// Only search into block-like nodes enclosing our "goto".
			// This prevents us from finding labels in nested blocks.
			case *ast.BlockStmt, *ast.CommClause, *ast.CaseClause:
				for _, p := range c.path {
					if n == p {
						return true
					}
				}
				return false
			case *ast.LabeledStmt:
				addLabel(highScore, n)
			}

			return true
		})
	}
}
