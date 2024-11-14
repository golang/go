// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types_test

import (
	"go/ast"
	"go/token"
	"strconv"
	"testing"
)

const (
	errorfMinArgCount = 4
	errorfFormatIndex = 2
)

// TestErrorCalls makes sure that check.errorf calls have at least
// errorfMinArgCount arguments (otherwise we should use check.error)
// and use balanced parentheses/brackets.
func TestErrorCalls(t *testing.T) {
	fset := token.NewFileSet()
	files, err := pkgFiles(fset, ".")
	if err != nil {
		t.Fatal(err)
	}

	for _, file := range files {
		ast.Inspect(file, func { n ->
			call, _ := n.(*ast.CallExpr)
			if call == nil {
				return true
			}
			selx, _ := call.Fun.(*ast.SelectorExpr)
			if selx == nil {
				return true
			}
			if !(isName(selx.X, "check") && isName(selx.Sel, "errorf")) {
				return true
			}
			// check.errorf calls should have at least errorfMinArgCount arguments:
			// position, code, format string, and arguments to format
			if n := len(call.Args); n < errorfMinArgCount {
				t.Errorf("%s: got %d arguments, want at least %d", fset.Position(call.Pos()), n, errorfMinArgCount)
				return false
			}
			format := call.Args[errorfFormatIndex]
			ast.Inspect(format, func { n ->
				if lit, _ := n.(*ast.BasicLit); lit != nil && lit.Kind == token.STRING {
					if s, err := strconv.Unquote(lit.Value); err == nil {
						if !balancedParentheses(s) {
							t.Errorf("%s: unbalanced parentheses/brackets", fset.Position(lit.ValuePos))
						}
					}
					return false
				}
				return true
			})
			return false
		})
	}
}

func isName(n ast.Node, name string) bool {
	if n, ok := n.(*ast.Ident); ok {
		return n.Name == name
	}
	return false
}

func balancedParentheses(s string) bool {
	var stack []byte
	for _, ch := range s {
		var open byte
		switch ch {
		case '(', '[', '{':
			stack = append(stack, byte(ch))
			continue
		case ')':
			open = '('
		case ']':
			open = '['
		case '}':
			open = '{'
		default:
			continue
		}
		// closing parenthesis/bracket must have matching opening
		top := len(stack) - 1
		if top < 0 || stack[top] != open {
			return false
		}
		stack = stack[:top]
	}
	return len(stack) == 0
}
