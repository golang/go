// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// A library for EBNF grammars. The input is text ([]byte) satisfying
// the following grammar (represented itself in EBNF):
//
//	Production  = name "=" Expression "." .
//	Expression  = Alternative { "|" Alternative } .
//	Alternative = Term { Term } .
//	Term        = name | token [ "..." token ] | Group | Option | Repetition .
//	Group       = "(" Expression ")" .
//	Option      = "[" Expression "]" .
//	Repetition  = "{" Expression "}" .
//
// A name is a Go identifier, a token is a Go string, and comments
// and white space follow the same rules as for the Go language.
// Production names starting with an uppercase Unicode letter denote
// non-terminal productions (i.e., productions which allow white-space
// and comments between tokens); all other production names denote
// lexical productions.
//
package ebnf

import (
	"container/vector";
	"fmt";
	"go/scanner";
	"go/token";
	"os";
	"strconv";
	"strings";
	"unicode";
	"utf8";
)


// ----------------------------------------------------------------------------
// Internal representation

type (
	// An Expression node represents a production expression.
	Expression interface {
		// Pos is the position of the first character of the syntactic construct
		Pos() token.Position;
	};

	// An Alternative node represents a non-empty list of alternative expressions.
	Alternative []Expression;  // x | y | z

	// A Sequence node represents a non-empty list of sequential expressions.
	Sequence []Expression;  // x y z

	// A Name node represents a production name.
	Name struct {
		token.Position;
		String string;
	};

	// A Token node represents a literal.
	Token struct {
		token.Position;
		String string;
	};

	// A List node represents a range of characters.
	Range struct {
		Begin, End *Token;  // begin ... end
	};

	// A Group node represents a grouped expression.
	Group struct {
		token.Position;
		Body Expression;  // (body)
	};

	// An Option node represents an optional expression.
	Option struct {
		token.Position;
		Body Expression;  // [body]
	};

	// A Repetition node represents a repeated expression.
	Repetition struct {
		token.Position;
		Body Expression;  // {body}
	};

	// A Production node represents an EBNF production.
	Production struct {
		Name *Name;
		Expr Expression;
	};

	// A Grammar is a set of EBNF productions. The map
	// is indexed by production name.
	//
	Grammar map [string] *Production;
)


func (x Alternative) Pos() token.Position {
	return x[0].Pos();  // the parser always generates non-empty Alternative
}


func (x Sequence) Pos() token.Position {
	return x[0].Pos();  // the parser always generates non-empty Sequences
}


func (x Range) Pos() token.Position {
	return x.Begin.Pos();
}


func (p *Production) Pos() token.Position {
	return p.Name.Pos();
}


// ----------------------------------------------------------------------------
// Error handling

// TODO(gri) This is the same code as in datafmt and go/parser.
//           Should factor this out as part of some parsing framework
//           that could also deal with reading various input sources.

// Error describes an individual error. The position Pos, if valid,
// indicates the format source position the error relates to. The
// error is specified with the Msg string.
//
type Error struct {
	Pos token.Position;
	Msg string;
}


// String returns the error message. If the error contains (line, column)
// position information, it starts with "line:column: ", otherwise it
// starts with a blank " ". 
//
func (e *Error) String() string {
	pos := " ";
	if e.Pos.IsValid() {
		pos = fmt.Sprintf("%d:%d: ", e.Pos.Line, e.Pos.Column);
	}
	return pos + e.Msg;
}


// An ErrorList is a list of errors encountered during parsing.
type ErrorList []*Error


// ErrorList implements SortInterface and the os.Error interface.

func (p ErrorList) Len() int  { return len(p); }
func (p ErrorList) Swap(i, j int)  { p[i], p[j] = p[j], p[i]; }
func (p ErrorList) Less(i, j int) bool  { return p[i].Pos.Offset < p[j].Pos.Offset; }


func (p ErrorList) String() string {
	switch len(p) {
	case 0:
		return "unspecified error";
	case 1:
		return p[0].String();
	}
	return fmt.Sprintf("%s (and %d more errors)", p[0].String(), len(p) - 1);
}


// ----------------------------------------------------------------------------
// Grammar verification

func isLexical(name string) bool {
	ch, len := utf8.DecodeRuneInString(name);
	return !unicode.IsUpper(ch);
}


type verifier struct {
	errors vector.Vector;
	worklist vector.Vector;
	reached Grammar;  // set of productions reached from (and including) the root production
	grammar Grammar;
}


func (v *verifier) error(pos token.Position, msg string) {
	v.errors.Push(&Error{pos, msg});
}


func makeErrorList(v *vector.Vector) os.Error {
	if v.Len() > 0 {
		errors := make(ErrorList, v.Len());
		for i := 0; i < v.Len(); i++ {
			errors[i] = v.At(i).(*Error);
		}
		return errors;
	}
	return nil;
}


func (v *verifier) push(prod *Production) {
	name := prod.Name.String;
	if _, found := v.reached[name]; !found {
		v.worklist.Push(prod);
		v.reached[name] = prod;
	}
}


func (v *verifier) verifyChar(x *Token) int {
	s := x.String;
	if utf8.RuneCountInString(s) != 1 {
		v.error(x.Pos(), "single char expected, found " + s);
		return 0;
	}
	ch, _ := utf8.DecodeRuneInString(s);
	return ch;
}


func (v *verifier) verifyExpr(expr Expression, lexical bool) {
	switch x := expr.(type) {
	case nil:
		// empty expression
	case Alternative:
		for _, e := range x {
			v.verifyExpr(e, lexical);
		}
	case Sequence:
		for _, e := range x {
			v.verifyExpr(e, lexical);
		}
	case *Name:
		// a production with this name must exist;
		// add it to the worklist if not yet processed
		if prod, found := v.grammar[x.String]; found {
			v.push(prod);
		} else {
			v.error(x.Pos(), "missing production " + x.String);
		}
		// within a lexical production references
		// to non-lexical productions are invalid
		if lexical && !isLexical(x.String) {
			v.error(x.Pos(), "reference to non-lexical production " + x.String);
		}
	case *Token:
		// nothing to do for now
	case *Range:
		i := v.verifyChar(x.Begin);
		j := v.verifyChar(x.End);
		if i >= j {
			v.error(x.Pos(), "decreasing character range");
		}
	case *Group:
		v.verifyExpr(x.Body, lexical);
	case *Option:
		v.verifyExpr(x.Body, lexical);
	case *Repetition:
		v.verifyExpr(x.Body, lexical);
	default:
		panic("unreachable");
	}
}


func (v *verifier) verify(grammar Grammar, start string) {
	// find root production
	root, found := grammar[start];
	if !found {
		var noPos token.Position;
		v.error(noPos, "no start production " + start);
		return;
	}

	// initialize verifier
	v.errors.Init(0);
	v.worklist.Init(0);
	v.reached = make(Grammar);
	v.grammar = grammar;

	// work through the worklist
	v.push(root);
	for v.worklist.Len() > 0 {
		prod := v.worklist.Pop().(*Production);
		v.verifyExpr(prod.Expr, isLexical(prod.Name.String));
	}

	// check if all productions were reached
	if len(v.reached) < len(v.grammar) {
		for name, prod := range v.grammar {
			if _, found := v.reached[name]; !found {
				v.error(prod.Pos(), name + " is unreachable");
			}
		}
	}
}


// Verify checks that:
//	- all productions used are defined
//	- all productions defined are used when beginning at start
//	- lexical productions refer only to other lexical productions
//
func Verify(grammar Grammar, start string) os.Error {
	var v verifier;
	v.verify(grammar, start);
	return makeErrorList(&v.errors);
}
