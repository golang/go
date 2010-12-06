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
	"go/scanner"
	"go/token"
	"os"
	"unicode"
	"utf8"
)


// ----------------------------------------------------------------------------
// Internal representation

type (
	// An Expression node represents a production expression.
	Expression interface {
		// Pos is the position of the first character of the syntactic construct
		Pos() token.Pos
	}

	// An Alternative node represents a non-empty list of alternative expressions.
	Alternative []Expression // x | y | z

	// A Sequence node represents a non-empty list of sequential expressions.
	Sequence []Expression // x y z

	// A Name node represents a production name.
	Name struct {
		StringPos token.Pos
		String    string
	}

	// A Token node represents a literal.
	Token struct {
		StringPos token.Pos
		String    string
	}

	// A List node represents a range of characters.
	Range struct {
		Begin, End *Token // begin ... end
	}

	// A Group node represents a grouped expression.
	Group struct {
		Lparen token.Pos
		Body   Expression // (body)
	}

	// An Option node represents an optional expression.
	Option struct {
		Lbrack token.Pos
		Body   Expression // [body]
	}

	// A Repetition node represents a repeated expression.
	Repetition struct {
		Lbrace token.Pos
		Body   Expression // {body}
	}

	// A Production node represents an EBNF production.
	Production struct {
		Name *Name
		Expr Expression
	}

	// A Grammar is a set of EBNF productions. The map
	// is indexed by production name.
	//
	Grammar map[string]*Production
)


func (x Alternative) Pos() token.Pos { return x[0].Pos() } // the parser always generates non-empty Alternative
func (x Sequence) Pos() token.Pos    { return x[0].Pos() } // the parser always generates non-empty Sequences
func (x *Name) Pos() token.Pos       { return x.StringPos }
func (x *Token) Pos() token.Pos      { return x.StringPos }
func (x *Range) Pos() token.Pos      { return x.Begin.Pos() }
func (x *Group) Pos() token.Pos      { return x.Lparen }
func (x *Option) Pos() token.Pos     { return x.Lbrack }
func (x *Repetition) Pos() token.Pos { return x.Lbrace }
func (x *Production) Pos() token.Pos { return x.Name.Pos() }


// ----------------------------------------------------------------------------
// Grammar verification

func isLexical(name string) bool {
	ch, _ := utf8.DecodeRuneInString(name)
	return !unicode.IsUpper(ch)
}


type verifier struct {
	fset *token.FileSet
	scanner.ErrorVector
	worklist []*Production
	reached  Grammar // set of productions reached from (and including) the root production
	grammar  Grammar
}


func (v *verifier) error(pos token.Pos, msg string) {
	v.Error(v.fset.Position(pos), msg)
}


func (v *verifier) push(prod *Production) {
	name := prod.Name.String
	if _, found := v.reached[name]; !found {
		v.worklist = append(v.worklist, prod)
		v.reached[name] = prod
	}
}


func (v *verifier) verifyChar(x *Token) int {
	s := x.String
	if utf8.RuneCountInString(s) != 1 {
		v.error(x.Pos(), "single char expected, found "+s)
		return 0
	}
	ch, _ := utf8.DecodeRuneInString(s)
	return ch
}


func (v *verifier) verifyExpr(expr Expression, lexical bool) {
	switch x := expr.(type) {
	case nil:
		// empty expression
	case Alternative:
		for _, e := range x {
			v.verifyExpr(e, lexical)
		}
	case Sequence:
		for _, e := range x {
			v.verifyExpr(e, lexical)
		}
	case *Name:
		// a production with this name must exist;
		// add it to the worklist if not yet processed
		if prod, found := v.grammar[x.String]; found {
			v.push(prod)
		} else {
			v.error(x.Pos(), "missing production "+x.String)
		}
		// within a lexical production references
		// to non-lexical productions are invalid
		if lexical && !isLexical(x.String) {
			v.error(x.Pos(), "reference to non-lexical production "+x.String)
		}
	case *Token:
		// nothing to do for now
	case *Range:
		i := v.verifyChar(x.Begin)
		j := v.verifyChar(x.End)
		if i >= j {
			v.error(x.Pos(), "decreasing character range")
		}
	case *Group:
		v.verifyExpr(x.Body, lexical)
	case *Option:
		v.verifyExpr(x.Body, lexical)
	case *Repetition:
		v.verifyExpr(x.Body, lexical)
	default:
		panic("unreachable")
	}
}


func (v *verifier) verify(fset *token.FileSet, grammar Grammar, start string) {
	// find root production
	root, found := grammar[start]
	if !found {
		// token.NoPos doesn't require a file set;
		// ok to set v.fset only afterwards
		v.error(token.NoPos, "no start production "+start)
		return
	}

	// initialize verifier
	v.fset = fset
	v.ErrorVector.Reset()
	v.worklist = v.worklist[0:0]
	v.reached = make(Grammar)
	v.grammar = grammar

	// work through the worklist
	v.push(root)
	for {
		n := len(v.worklist) - 1
		if n < 0 {
			break
		}
		prod := v.worklist[n]
		v.worklist = v.worklist[0:n]
		v.verifyExpr(prod.Expr, isLexical(prod.Name.String))
	}

	// check if all productions were reached
	if len(v.reached) < len(v.grammar) {
		for name, prod := range v.grammar {
			if _, found := v.reached[name]; !found {
				v.error(prod.Pos(), name+" is unreachable")
			}
		}
	}
}


// Verify checks that:
//	- all productions used are defined
//	- all productions defined are used when beginning at start
//	- lexical productions refer only to other lexical productions
//
// Position information is interpreted relative to the file set fset.
//
func Verify(fset *token.FileSet, grammar Grammar, start string) os.Error {
	var v verifier
	v.verify(fset, grammar, start)
	return v.GetError(scanner.Sorted)
}
