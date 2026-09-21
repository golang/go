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

	pos, _ := walkStringLiteral(lit, lit.End(), offset)
	return pos, nil
}

// OffsetInStringLiteral returns the byte offset within the logical (unquoted)
// string corresponding to the specified source position.
func OffsetInStringLiteral(lit *ast.BasicLit, pos token.Pos) (int, error) {
	if !NodeContainsPos(lit, pos) {
		return 0, fmt.Errorf("invalid position")
	}

	raw := lit.Value

	value, err := strconv.Unquote(raw)
	if err != nil {
		return 0, err
	}

	_, offset := walkStringLiteral(lit, pos, len(value))
	return offset, nil
}

// walkStringLiteral iterates through the raw string literal to map between
// a file position and a logical byte offset. It stops when it reaches
// either the targetPos or the targetOffset.
//
// TODO(hxjiang): consider making an iterator.
func walkStringLiteral(lit *ast.BasicLit, targetPos token.Pos, targetOffset int) (token.Pos, int) {
	raw := lit.Value
	norm := int(lit.End()-lit.Pos()) > len(lit.Value)

	// remove quotes
	quote := raw[0] // '"' or '`'
	raw = raw[1 : len(raw)-1]

	var (
		i   = 0             // byte index within logical value
		pos = lit.Pos() + 1 // position within literal
	)

	for raw != "" {
		r, _, rest, _ := strconv.UnquoteChar(raw, quote) // can't fail
		sz := len(raw) - len(rest)                       // length of literal char in raw bytes

		nextPos := pos + token.Pos(sz)
		if norm && r == '\n' {
			nextPos++
		}
		nextI := i + utf8.RuneLen(r) // length of logical char in "cooked" bytes

		if nextPos > targetPos || nextI > targetOffset {
			break
		}

		raw = raw[sz:]
		i = nextI
		pos = nextPos
	}

	return pos, i
}
