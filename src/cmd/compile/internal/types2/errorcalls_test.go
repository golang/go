// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package types2_test

import (
	"cmd/compile/internal/syntax"
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
	files, err := pkgFiles(".")
	if err != nil {
		t.Fatal(err)
	}

	for _, file := range files {
		syntax.Inspect(file, func(n syntax.Node) bool {
			call, _ := n.(*syntax.CallExpr)
			if call == nil {
				return true
			}
			selx, _ := call.Fun.(*syntax.SelectorExpr)
			if selx == nil {
				return true
			}
			if !(isName(selx.X, "check") && isName(selx.Sel, "errorf")) {
				return true
			}
			// check.errorf calls should have at least errorfMinArgCount arguments:
			// position, code, format string, and arguments to format
			if n := len(call.ArgList); n < errorfMinArgCount {
				t.Errorf("%s: got %d arguments, want at least %d", call.Pos(), n, errorfMinArgCount)
				return false
			}
			format := call.ArgList[errorfFormatIndex]
			syntax.Inspect(format, func(n syntax.Node) bool {
				if lit, _ := n.(*syntax.BasicLit); lit != nil && lit.Kind == syntax.StringLit {
					if s, err := strconv.Unquote(lit.Value); err == nil {
						if !balancedParentheses(s) {
							t.Errorf("%s: unbalanced parentheses/brackets", lit.Pos())
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

func isName(n syntax.Node, name string) bool {
	if n, ok := n.(*syntax.Name); ok {
		return n.Value == name
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
