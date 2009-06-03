// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package ast

import (
	"datafmt";
	"go/ast";
	"go/token";
	"io";
	"os";
)


// Format is a customized datafmt.Format for printing of ASTs.
type Format datafmt.Format;


// ----------------------------------------------------------------------------
// Custom formatters

// The AST-specific formatting state is maintained by a state variable.
type state struct {
	// for now we have very little state
	// TODO maintain list of unassociated comments
	optSemi *bool
}


func (s *state) Copy() datafmt.Environment {
	optSemi := *s.optSemi;
	return &state{&optSemi};
}


func isValidPos(s *datafmt.State, value interface{}, ruleName string) bool {
	pos := value.(token.Position);
	return pos.IsValid();
}


func isSend(s *datafmt.State, value interface{}, ruleName string) bool {
	return value.(ast.ChanDir) & ast.SEND != 0;
}


func isRecv(s *datafmt.State, value interface{}, ruleName string) bool {
	return value.(ast.ChanDir) & ast.RECV != 0;
}


func isMultiLineComment(s *datafmt.State, value interface{}, ruleName string) bool {
	return value.([]byte)[1] == '*';
}


func clearOptSemi(s *datafmt.State, value interface{}, ruleName string) bool {
	*s.Env().(*state).optSemi = false;
	return true;
}


func setOptSemi(s *datafmt.State, value interface{}, ruleName string) bool {
	*s.Env().(*state).optSemi = true;
	return true;
}


func optSemi(s *datafmt.State, value interface{}, ruleName string) bool {
	if !*s.Env().(*state).optSemi {
		s.Write([]byte{';'});
	}
	return true;
}


var fmap = datafmt.FormatterMap {
	"isValidPos": isValidPos,
	"isSend": isSend,
	"isRecv": isRecv,
	"isMultiLineComment": isMultiLineComment,
	"/": clearOptSemi,
	"clearOptSemi": clearOptSemi,
	"setOptSemi": setOptSemi,
	"optSemi": optSemi,
}


// ----------------------------------------------------------------------------
// Printing

// NewFormat parses a datafmt format specification from a file
// and adds AST-specific custom formatter rules. The result is
// the customized format or an os.Error, if any.
//
func NewFormat(filename string) (Format, os.Error) {
	src, err := io.ReadFile(filename);
	if err != nil {
		return nil, err;
	}
	f, err := datafmt.Parse(src, fmap);
	return Format(f), err;
}


// Fprint formats each AST node provided as argument according to the
// format f and writes to standard output. The result is the total number
// of bytes written and an os.Error, if any.
//
func (f Format) Fprint(w io.Writer, nodes ...) (int, os.Error) {
	s := state{new(bool)};
	return datafmt.Format(f).Fprint(w, &s, nodes);
}


// Fprint formats each AST node provided as argument according to the
// format f and writes to w. The result is the total number of bytes
// written and an os.Error, if any.
//
func (f Format) Print(nodes ...) (int, os.Error) {
	return f.Fprint(os.Stdout, nodes);
}
