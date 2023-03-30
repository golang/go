// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package safetoken provides wrappers around methods in go/token,
// that return errors rather than panicking.
//
// It also provides a central place for workarounds in the underlying
// packages. The use of this package's functions instead of methods of
// token.File (such as Offset, Position, and PositionFor) is mandatory
// throughout the gopls codebase and enforced by a static check.
package safetoken

import (
	"fmt"
	"go/token"
)

// Offset returns f.Offset(pos), but first checks that the file
// contains the pos.
//
// The definition of "contains" here differs from that of token.File
// in order to work around a bug in the parser (issue #57490): during
// error recovery, the parser may create syntax nodes whose computed
// End position is 1 byte beyond EOF, which would cause
// token.File.Offset to panic. The workaround is that this function
// accepts a Pos that is exactly 1 byte beyond EOF and maps it to the
// EOF offset.
func Offset(f *token.File, pos token.Pos) (int, error) {
	if !inRange(f, pos) {
		// Accept a Pos that is 1 byte beyond EOF,
		// and map it to the EOF offset.
		// (Workaround for #57490.)
		if int(pos) == f.Base()+f.Size()+1 {
			return f.Size(), nil
		}

		return -1, fmt.Errorf("pos %d is not in range [%d:%d] of file %s",
			pos, f.Base(), f.Base()+f.Size(), f.Name())
	}
	return int(pos) - f.Base(), nil
}

// Offsets returns Offset(start) and Offset(end).
func Offsets(f *token.File, start, end token.Pos) (int, int, error) {
	startOffset, err := Offset(f, start)
	if err != nil {
		return 0, 0, fmt.Errorf("start: %v", err)
	}
	endOffset, err := Offset(f, end)
	if err != nil {
		return 0, 0, fmt.Errorf("end: %v", err)
	}
	return startOffset, endOffset, nil
}

// Pos returns f.Pos(offset), but first checks that the offset is
// non-negative and not larger than the size of the file.
func Pos(f *token.File, offset int) (token.Pos, error) {
	if !(0 <= offset && offset <= f.Size()) {
		return token.NoPos, fmt.Errorf("offset %d is not in range for file %s of size %d", offset, f.Name(), f.Size())
	}
	return token.Pos(f.Base() + offset), nil
}

// inRange reports whether file f contains position pos,
// according to the invariants of token.File.
//
// This function is not public because of the ambiguity it would
// create w.r.t. the definition of "contains". Use Offset instead.
func inRange(f *token.File, pos token.Pos) bool {
	return token.Pos(f.Base()) <= pos && pos <= token.Pos(f.Base()+f.Size())
}

// Position returns the Position for the pos value in the given file.
//
// p must be NoPos, a valid Pos in the range of f, or exactly 1 byte
// beyond the end of f. (See [Offset] for explanation.)
// Any other value causes a panic.
//
// Line directives (//line comments) are ignored.
func Position(f *token.File, pos token.Pos) token.Position {
	// Work around issue #57490.
	if int(pos) == f.Base()+f.Size()+1 {
		pos--
	}

	// TODO(adonovan): centralize the workaround for
	// golang/go#41029 (newline at EOF) here too.

	return f.PositionFor(pos, false)
}

// Line returns the line number for the given offset in the given file.
func Line(f *token.File, pos token.Pos) int {
	return Position(f, pos).Line
}

// StartPosition converts a start Pos in the FileSet into a Position.
//
// Call this function only if start represents the start of a token or
// parse tree, such as the result of Node.Pos().  If start is the end of
// an interval, such as Node.End(), call EndPosition instead, as it
// may need the correction described at [Position].
func StartPosition(fset *token.FileSet, start token.Pos) (_ token.Position) {
	if f := fset.File(start); f != nil {
		return Position(f, start)
	}
	return
}

// EndPosition converts an end Pos in the FileSet into a Position.
//
// Call this function only if pos represents the end of
// a non-empty interval, such as the result of Node.End().
func EndPosition(fset *token.FileSet, end token.Pos) (_ token.Position) {
	if f := fset.File(end); f != nil && int(end) > f.Base() {
		return Position(f, end)
	}

	// Work around issue #57490.
	if f := fset.File(end - 1); f != nil {
		return Position(f, end)
	}

	return
}
