// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

import "fmt"

type token uint

const (
	_ token = iota
	_EOF

	// names and literals
	_Name
	_Literal

	// operators and operations
	_Operator // excluding '*' (_Star)
	_AssignOp
	_IncOp
	_Assign
	_Define
	_Arrow
	_Star

	// delimitors
	_Lparen
	_Lbrack
	_Lbrace
	_Rparen
	_Rbrack
	_Rbrace
	_Comma
	_Semi
	_Colon
	_Dot
	_DotDotDot

	// keywords
	_Break
	_Case
	_Chan
	_Const
	_Continue
	_Default
	_Defer
	_Else
	_Fallthrough
	_For
	_Func
	_Go
	_Goto
	_If
	_Import
	_Interface
	_Map
	_Package
	_Range
	_Return
	_Select
	_Struct
	_Switch
	_Type
	_Var

	tokenCount
)

const (
	// for BranchStmt
	Break       = _Break
	Continue    = _Continue
	Fallthrough = _Fallthrough
	Goto        = _Goto

	// for CallStmt
	Go    = _Go
	Defer = _Defer
)

var tokstrings = [...]string{
	// source control
	_EOF: "EOF",

	// names and literals
	_Name:    "name",
	_Literal: "literal",

	// operators and operations
	_Operator: "op",
	_AssignOp: "op=",
	_IncOp:    "opop",
	_Assign:   "=",
	_Define:   ":=",
	_Arrow:    "<-",
	_Star:     "*",

	// delimitors
	_Lparen:    "(",
	_Lbrack:    "[",
	_Lbrace:    "{",
	_Rparen:    ")",
	_Rbrack:    "]",
	_Rbrace:    "}",
	_Comma:     ",",
	_Semi:      ";",
	_Colon:     ":",
	_Dot:       ".",
	_DotDotDot: "...",

	// keywords
	_Break:       "break",
	_Case:        "case",
	_Chan:        "chan",
	_Const:       "const",
	_Continue:    "continue",
	_Default:     "default",
	_Defer:       "defer",
	_Else:        "else",
	_Fallthrough: "fallthrough",
	_For:         "for",
	_Func:        "func",
	_Go:          "go",
	_Goto:        "goto",
	_If:          "if",
	_Import:      "import",
	_Interface:   "interface",
	_Map:         "map",
	_Package:     "package",
	_Range:       "range",
	_Return:      "return",
	_Select:      "select",
	_Struct:      "struct",
	_Switch:      "switch",
	_Type:        "type",
	_Var:         "var",
}

func (tok token) String() string {
	var s string
	if 0 <= tok && int(tok) < len(tokstrings) {
		s = tokstrings[tok]
	}
	if s == "" {
		s = fmt.Sprintf("<tok-%d>", tok)
	}
	return s
}

// Make sure we have at most 64 tokens so we can use them in a set.
const _ uint64 = 1 << (tokenCount - 1)

// contains reports whether tok is in tokset.
func contains(tokset uint64, tok token) bool {
	return tokset&(1<<tok) != 0
}

type LitKind uint

const (
	IntLit LitKind = iota
	FloatLit
	ImagLit
	RuneLit
	StringLit
)

type Operator uint

const (
	_    Operator = iota
	Def           // :=
	Not           // !
	Recv          // <-

	// precOrOr
	OrOr // ||

	// precAndAnd
	AndAnd // &&

	// precCmp
	Eql // ==
	Neq // !=
	Lss // <
	Leq // <=
	Gtr // >
	Geq // >=

	// precAdd
	Add // +
	Sub // -
	Or  // |
	Xor // ^

	// precMul
	Mul    // *
	Div    // /
	Rem    // %
	And    // &
	AndNot // &^
	Shl    // <<
	Shr    // >>
)

var opstrings = [...]string{
	// prec == 0
	Def:  ":", // : in :=
	Not:  "!",
	Recv: "<-",

	// precOrOr
	OrOr: "||",

	// precAndAnd
	AndAnd: "&&",

	// precCmp
	Eql: "==",
	Neq: "!=",
	Lss: "<",
	Leq: "<=",
	Gtr: ">",
	Geq: ">=",

	// precAdd
	Add: "+",
	Sub: "-",
	Or:  "|",
	Xor: "^",

	// precMul
	Mul:    "*",
	Div:    "/",
	Rem:    "%",
	And:    "&",
	AndNot: "&^",
	Shl:    "<<",
	Shr:    ">>",
}

func (op Operator) String() string {
	var s string
	if 0 <= op && int(op) < len(opstrings) {
		s = opstrings[op]
	}
	if s == "" {
		s = fmt.Sprintf("<op-%d>", op)
	}
	return s
}

// Operator precedences
const (
	_ = iota
	precOrOr
	precAndAnd
	precCmp
	precAdd
	precMul
)
