// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package safetoken

import "go/token"

// Range represents a source code range in token.Pos form.
//
// It also carries the token.File that produced the position,
// so that it is capable of returning (file, line, col8) information.
// However it cannot be converted to protocol (UTF-16) form
// without access to file content; to do that, use a protocol.ContentMapper.
//
// TODO(adonovan): Eliminate most/all uses of Range in gopls, as
// without a Mapper it's not really self-contained.  It is mostly used
// by completion. Given access to complete.mapper, it could use a pair
// of byte offsets instead.
type Range struct {
	TokFile    *token.File // non-nil
	Start, End token.Pos   // both IsValid()
}

// NewRange creates a new Range from a token.File and two positions within it.
// The given start position must be valid; if end is invalid, start is used as
// the end position.
//
// (If you only have a token.FileSet, use file = fset.File(start). But
// most callers know exactly which token.File they're dealing with and
// should pass it explicitly. Not only does this save a lookup, but it
// brings us a step closer to eliminating the global FileSet.)
func NewRange(file *token.File, start, end token.Pos) Range {
	if file == nil {
		panic("nil *token.File")
	}
	if !start.IsValid() {
		panic("invalid start token.Pos")
	}
	if !end.IsValid() {
		end = start
	}

	// TODO(adonovan): ideally we would make this stronger assertion:
	//
	//   // Assert that file is non-nil and contains start and end.
	//   _ = file.Offset(start)
	//   _ = file.Offset(end)
	//
	// but some callers (e.g. packageCompletionSurrounding) don't ensure this precondition.

	return Range{
		TokFile: file,
		Start:   start,
		End:     end,
	}
}

// IsPoint returns true if the range represents a single point.
func (r Range) IsPoint() bool {
	return r.Start == r.End
}
