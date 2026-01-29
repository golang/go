// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package astutil provides various AST utility functions for gopls.
package astutil

import (
	"bytes"
	"go/scanner"
	"go/token"
)

// PurgeFuncBodies returns a copy of src in which the contents of each
// outermost {...} region except struct and interface types have been
// deleted. This reduces the amount of work required to parse the
// top-level declarations.
//
// PurgeFuncBodies does not preserve newlines or position information.
// Also, if the input is invalid, parsing the output of
// PurgeFuncBodies may result in a different tree due to its effects
// on parser error recovery.
func PurgeFuncBodies(src []byte) []byte {
	// Destroy the content of any {...}-bracketed regions that are
	// not immediately preceded by a "struct" or "interface"
	// token.  That includes function bodies, composite literals,
	// switch/select bodies, and all blocks of statements.
	// This will lead to non-void functions that don't have return
	// statements, which of course is a type error, but that's ok.

	var out bytes.Buffer
	file := token.NewFileSet().AddFile("", -1, len(src))
	var sc scanner.Scanner
	sc.Init(file, src, nil, 0)
	var prev token.Token
	var cursor int         // last consumed src offset
	var braces []token.Pos // stack of unclosed braces or -1 for struct/interface type
	for {
		pos, tok, _ := sc.Scan()
		if tok == token.EOF {
			break
		}
		switch tok {
		case token.COMMENT:
			// TODO(adonovan): opt: skip, to save an estimated 20% of time.

		case token.LBRACE:
			if prev == token.STRUCT || prev == token.INTERFACE {
				pos = -1
			}
			braces = append(braces, pos)

		case token.RBRACE:
			if last := len(braces) - 1; last >= 0 {
				top := braces[last]
				braces = braces[:last]
				if top < 0 {
					// struct/interface type: leave alone
				} else if len(braces) == 0 { // toplevel only
					// Delete {...} body.
					start := file.Offset(top)
					end := file.Offset(pos)
					out.Write(src[cursor : start+len("{")])
					cursor = end
				}
			}
		}
		prev = tok
	}
	out.Write(src[cursor:])
	return out.Bytes()
}
