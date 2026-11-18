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
// outermost {...} region have been deleted, except for struct and
// interface type bodies and the bodies of length-elided array
// literals ([...]T), whose element count is part of the type. It
// includes function bodies, function-literal bodies, and the bodies
// of slice, map, and explicitly-sized array composite literals (whose
// contents don't affect the type of the enclosing declaration). This
// reduces the amount of work required to parse the top-level
// declarations.
//
// PurgeFuncBodies does not preserve newlines or position information.
// Also, if the input is invalid, parsing the output of
// PurgeFuncBodies may result in a different tree due to its effects
// on parser error recovery.
func PurgeFuncBodies(src []byte) []byte {
	if len(src) == 0 {
		return nil
	}

	// Destroy the content of any {...}-bracketed regions that are
	// not immediately preceded by a "struct" or "interface" token,
	// and that are not the body of a length-elided array literal.
	// That includes function bodies, switch/select bodies, and most
	// composite literals; this will lead to non-void functions that
	// don't have return statements, which of course is a type error,
	// but that's ok.

	file := token.NewFileSet().AddFile("", -1, len(src))
	// Allocate the scanner's line table once, at its final size.
	file.SetLines(make([]int, 1, 1+bytes.Count(src, []byte("\n"))))
	var sc scanner.Scanner
	sc.Init(file, src, nil, 0)
	var prev token.Token
	var braces []token.Pos // stack of unclosed braces, or -1 for a region we preserve
	var ellipsis bool      // saw "[...]" not yet consumed by a literal-body "{"
	var cuts []int         // start and end offsets of each deleted region
	deleted := 0           // total length of deleted regions
	for {
		pos, tok, _ := sc.Scan()
		if tok == token.EOF {
			break
		}
		switch tok {
		case token.COMMENT:
			// TODO(adonovan): opt: skip, to save an estimated 20% of time.

		case token.SEMICOLON:
			ellipsis = false

		case token.RBRACK:
			// "...]" occurs only in the array-type prefix of a
			// composite literal; variadic "..." is followed by
			// a type or ")", never "]".
			if prev == token.ELLIPSIS {
				ellipsis = true
			}

		case token.LBRACE:
			if prev == token.STRUCT || prev == token.INTERFACE {
				pos = -1 // type body: preserve (don't consume ellipsis)
			} else if ellipsis {
				pos = -1 // [...]T literal body: preserve
				ellipsis = false
			}
			braces = append(braces, pos)

		case token.RBRACE:
			if last := len(braces) - 1; last >= 0 {
				top := braces[last]
				braces = braces[:last]
				if top < 0 {
					// preserve
				} else if len(braces) == 0 { // toplevel only
					// Delete {...} body.
					start := file.Offset(top) + len("{")
					end := file.Offset(pos)
					cuts = append(cuts, start, end)
					deleted += end - start
				}
			}
		}
		prev = tok
	}

	// Copy the kept regions into a slice of the exact size,
	// so we avoid growth-by-doubling for these long-lived arrays.
	out := make([]byte, 0, len(src)-deleted)
	cursor := 0 // last consumed src offset
	for i := 0; i < len(cuts); i += 2 {
		out = append(out, src[cursor:cuts[i]]...)
		cursor = cuts[i+1]
	}
	return append(out, src[cursor:]...)
}
