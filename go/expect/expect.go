// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package expect provides support for interpreting structured comments in Go
source code as test expectations.

This is primarily intended for writing tests of things that process Go source
files, although it does not directly depend on the testing package.

Collect notes with the Extract or Parse functions, and use the
MatchBefore function to find matches within the lines the comments were on.

The interpretation of the notes depends on the application.
For example, the test suite for a static checking tool might
use a @diag note to indicate an expected diagnostic:

   fmt.Printf("%s", 1) //@ diag("%s wants a string, got int")

By contrast, the test suite for a source code navigation tool
might use notes to indicate the positions of features of
interest, the actions to be performed by the test,
and their expected outcomes:

   var x = 1 //@ x_decl
   ...
   print(x) //@ definition("x", x_decl)
   print(x) //@ typeof("x", "int")


Note comment syntax

Note comments always start with the special marker @, which must be the
very first character after the comment opening pair, so //@ or /*@ with no
spaces.

This is followed by a comma separated list of notes.

A note always starts with an identifier, which is optionally followed by an
argument list. The argument list is surrounded with parentheses and contains a
comma-separated list of arguments.
The empty parameter list and the missing parameter list are distinguishable if
needed; they result in a nil or an empty list in the Args parameter respectively.

Arguments are either identifiers or literals.
The literals supported are the basic value literals, of string, float, integer
true, false or nil. All the literals match the standard go conventions, with
all bases of integers, and both quote and backtick strings.
There is one extra literal type, which is a string literal preceded by the
identifier "re" which is compiled to a regular expression.
*/
package expect

import (
	"bytes"
	"fmt"
	"go/token"
	"regexp"
)

// Note is a parsed note from an expect comment.
// It knows the position of the start of the comment, and the name and
// arguments that make up the note.
type Note struct {
	Pos  token.Pos     // The position at which the note identifier appears
	Name string        // the name associated with the note
	Args []interface{} // the arguments for the note
}

// ReadFile  is the type of a function that can provide file contents for a
// given filename.
// This is used in MatchBefore to look up the content of the file in order to
// find the line to match the pattern against.
type ReadFile func(filename string) ([]byte, error)

// MatchBefore attempts to match a pattern in the line before the supplied pos.
// It uses the FileSet and the ReadFile to work out the contents of the line
// that end is part of, and then matches the pattern against the content of the
// start of that line up to the supplied position.
// The pattern may be either a simple string, []byte or a *regexp.Regexp.
// MatchBefore returns the range of the line that matched the pattern, and
// invalid positions if there was no match, or an error if the line could not be
// found.
func MatchBefore(fset *token.FileSet, readFile ReadFile, end token.Pos, pattern interface{}) (token.Pos, token.Pos, error) {
	f := fset.File(end)
	content, err := readFile(f.Name())
	if err != nil {
		return token.NoPos, token.NoPos, fmt.Errorf("invalid file: %v", err)
	}
	position := f.Position(end)
	startOffset := f.Offset(lineStart(f, position.Line))
	endOffset := f.Offset(end)
	line := content[startOffset:endOffset]
	matchStart, matchEnd := -1, -1
	switch pattern := pattern.(type) {
	case string:
		bytePattern := []byte(pattern)
		matchStart = bytes.Index(line, bytePattern)
		if matchStart >= 0 {
			matchEnd = matchStart + len(bytePattern)
		}
	case []byte:
		matchStart = bytes.Index(line, pattern)
		if matchStart >= 0 {
			matchEnd = matchStart + len(pattern)
		}
	case *regexp.Regexp:
		match := pattern.FindIndex(line)
		if len(match) > 0 {
			matchStart = match[0]
			matchEnd = match[1]
		}
	}
	if matchStart < 0 {
		return token.NoPos, token.NoPos, nil
	}
	return f.Pos(startOffset + matchStart), f.Pos(startOffset + matchEnd), nil
}

// this functionality was borrowed from the analysisutil package
func lineStart(f *token.File, line int) token.Pos {
	// Use binary search to find the start offset of this line.
	//
	// TODO(adonovan): eventually replace this function with the
	// simpler and more efficient (*go/token.File).LineStart, added
	// in go1.12.

	min := 0        // inclusive
	max := f.Size() // exclusive
	for {
		offset := (min + max) / 2
		pos := f.Pos(offset)
		posn := f.Position(pos)
		if posn.Line == line {
			return pos - (token.Pos(posn.Column) - 1)
		}

		if min+1 >= max {
			return token.NoPos
		}

		if posn.Line < line {
			min = offset
		} else {
			max = offset
		}
	}
}
