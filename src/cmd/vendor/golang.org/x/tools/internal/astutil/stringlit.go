// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package astutil

import (
	"fmt"
	"go/ast"
	"go/token"
	"strconv"
	"unicode/utf8"
)

// RangeInStringLiteral calculates the positional range within a string literal
// corresponding to the specified start and end byte offsets within the logical string.
func RangeInStringLiteral(lit *ast.BasicLit, start, end int) (Range, error) {
	startPos, err := PosInStringLiteral(lit, start)
	if err != nil {
		return Range{}, fmt.Errorf("start: %v", err)
	}
	endPos, err := PosInStringLiteral(lit, end)
	if err != nil {
		return Range{}, fmt.Errorf("end: %v", err)
	}
	return Range{startPos, endPos}, nil
}

// PosInStringLiteral returns the position within a string literal
// corresponding to the specified byte offset within the logical
// string that it denotes.
func PosInStringLiteral(lit *ast.BasicLit, offset int) (token.Pos, error) {
	raw := lit.Value

	value, err := strconv.Unquote(raw)
	if err != nil {
		return 0, err
	}
	if !(0 <= offset && offset <= len(value)) {
		return 0, fmt.Errorf("invalid offset")
	}

	// remove quotes
	quote := raw[0] // '"' or '`'
	raw = raw[1 : len(raw)-1]

	var (
		i   = 0                // byte index within logical value
		pos = lit.ValuePos + 1 // position within literal
	)
	for raw != "" && i < offset {
		r, _, rest, _ := strconv.UnquoteChar(raw, quote) // can't fail
		sz := len(raw) - len(rest)                       // length of literal char in raw bytes
		pos += token.Pos(sz)
		raw = raw[sz:]
		i += utf8.RuneLen(r)
	}
	return pos, nil
}
