// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import (
	"strings"
	"testing"
)

const logosConstIfEndSrc = `package p

const debugOn = true

func f() int {
	n := 0
	const if debugOn {
		n = 1
	}
	const if false {
		n = 2
	} else const if debugOn {
		n = 3
	} else {
		n = 4
	}
	return n
}
`

// TestLogosConstIfEndPos requires EndPos of a `const if` to be one past
// the closing brace of its last block, for a lone statement and for an
// else-const-if chain, so a splice over the statement never takes the
// newline that follows it. It also requires a name before the block to
// parse as the condition (`const if name {`).
func TestLogosConstIfEndPos(t *testing.T) {
	var errs []string
	file, _ := Parse(NewFileBase("end.lgo"), strings.NewReader(logosConstIfEndSrc), func(err error) { errs = append(errs, err.Error()) }, nil, LogosMode)
	if len(errs) > 0 {
		t.Fatalf("parse errors: %v", errs)
	}
	lines := strings.Split(logosConstIfEndSrc, "\n")
	for _, decl := range file.DeclList {
		fn, ok := decl.(*FuncDecl)
		if !ok {
			continue
		}
		for _, s := range fn.Body.List {
			if c, ok := s.(*ConstIfStmt); ok {
				end := EndPos(c)
				if col, want := int(end.Col())-1, len(lines[end.Line()-1]); col != want {
					t.Errorf("const if at %s ends at offset %d of its last line, want %d", c.Pos(), col, want)
				}
			}
		}
	}
}
