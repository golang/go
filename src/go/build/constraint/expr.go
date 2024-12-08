// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package constraint implements parsing and evaluation of build constraint lines.
// See https://golang.org/cmd/go/#hdr-Build_constraints for documentation about build constraints themselves.
//
// This package parses both the original “// +build” syntax and the “//go:build” syntax that was added in Go 1.17.
// See https://golang.org/design/draft-gobuild for details about the “//go:build” syntax.
package constraint

import (
	"errors"
	"strings"
	"unicode"
	"unicode/utf8"
)

// maxSize is a limit used to control the complexity of expressions, in order
// to prevent stack exhaustion issues due to recursion.
const maxSize = 1000

// An Expr is a build tag constraint expression.
// The underlying concrete type is *[AndExpr], *[OrExpr], *[NotExpr], or *[TagExpr].
type Expr interface {
	// String returns the string form of the expression,
	// using the boolean syntax used in //go:build lines.
	String() string

	// Eval reports whether the expression evaluates to true.
	// It calls ok(tag) as needed to find out whether a given build tag
	// is satisfied by the current build configuration.
	Eval(ok func(tag string) bool) bool

	// The presence of an isExpr method explicitly marks the type as an Expr.
	// Only implementations in this package should be used as Exprs.
	isExpr()
}

// A TagExpr is an [Expr] for the single tag Tag.
type TagExpr struct {
	Tag string // for example, “linux” or “cgo”
}

func (x *TagExpr) isExpr() {}

func (x *TagExpr) Eval(ok func(tag string) bool) bool {
	return ok(x.Tag)
}

func (x *TagExpr) String() string {
	return x.Tag
}

func tag(tag string) Expr { return &TagExpr{tag} }

// A NotExpr represents the expression !X (the negation of X).
type NotExpr struct {
	X Expr
}

func (x *NotExpr) isExpr() {}

func (x *NotExpr) Eval(ok func(tag string) bool) bool {
	return !x.X.Eval(ok)
}

func (x *NotExpr) String() string {
	s := x.X.String()
	switch x.X.(type) {
	case *AndExpr, *OrExpr:
		s = "(" + s + ")"
	}
	return "!" + s
}

func not(x Expr) Expr { return &NotExpr{x} }

// An AndExpr represents the expression X && Y.
type AndExpr struct {
	X, Y Expr
}

func (x *AndExpr) isExpr() {}

func (x *AndExpr) Eval(ok func(tag string) bool) bool {
	// Note: Eval both, to make sure ok func observes all tags.
	xok := x.X.Eval(ok)
	yok := x.Y.Eval(ok)
	return xok && yok
}

func (x *AndExpr) String() string {
	return andArg(x.X) + " && " + andArg(x.Y)
}

func andArg(x Expr) string {
	s := x.String()
	if _, ok := x.(*OrExpr); ok {
		s = "(" + s + ")"
	}
	return s
}

func and(x, y Expr) Expr {
	return &AndExpr{x, y}
}

// An OrExpr represents the expression X || Y.
type OrExpr struct {
	X, Y Expr
}

func (x *OrExpr) isExpr() {}

func (x *OrExpr) Eval(ok func(tag string) bool) bool {
	// Note: Eval both, to make sure ok func observes all tags.
	xok := x.X.Eval(ok)
	yok := x.Y.Eval(ok)
	return xok || yok
}

func (x *OrExpr) String() string {
	return orArg(x.X) + " || " + orArg(x.Y)
}

func orArg(x Expr) string {
	s := x.String()
	if _, ok := x.(*AndExpr); ok {
		s = "(" + s + ")"
	}
	return s
}

func or(x, y Expr) Expr {
	return &OrExpr{x, y}
}

// A SyntaxError reports a syntax error in a parsed build expression.
type SyntaxError struct {
	Offset int    // byte offset in input where error was detected
	Err    string // description of error
}

func (e *SyntaxError) Error() string {
	return e.Err
}

var errNotConstraint = errors.New("not a build constraint")

// Parse parses a single build constraint line of the form “//go:build ...” or “// +build ...”
// and returns the corresponding boolean expression.
func Parse(line string) (Expr, error) {
	if text, ok := splitGoBuild(line); ok {
		return parseExpr(text)
	}
	if text, ok := splitPlusBuild(line); ok {
		return parsePlusBuildExpr(text)
	}
	return nil, errNotConstraint
}

// IsGoBuild reports whether the line of text is a “//go:build” constraint.
// It only checks the prefix of the text, not that the expression itself parses.
func IsGoBuild(line string) bool {
	_, ok := splitGoBuild(line)
	return ok
}

// splitGoBuild splits apart the leading //go:build prefix in line from the build expression itself.
// It returns "", false if the input is not a //go:build line or if the input contains multiple lines.
func splitGoBuild(line string) (expr string, ok bool) {
	// A single trailing newline is OK; otherwise multiple lines are not.
	if len(line) > 0 && line[len(line)-1] == '\n' {
		line = line[:len(line)-1]
	}
	if strings.Contains(line, "\n") {
		return "", false
	}

	if !strings.HasPrefix(line, "//go:build") {
		return "", false
	}

	line = strings.TrimSpace(line)
	line = line[len("//go:build"):]

	// If strings.TrimSpace finds more to trim after removing the //go:build prefix,
	// it means that the prefix was followed by a space, making this a //go:build line
	// (as opposed to a //go:buildsomethingelse line).
	// If line is empty, we had "//go:build" by itself, which also counts.
	trim := strings.TrimSpace(line)
	if len(line) == len(trim) && line != "" {
		return "", false
	}

	return trim, true
}

// An exprParser holds state for parsing a build expression.
type exprParser struct {
	s string // input string
	i int    // next read location in s

	tok   string // last token read
	isTag bool
	pos   int // position (start) of last token

	size int
}

// parseExpr parses a boolean build tag expression.
func parseExpr(text string) (x Expr, err error) {
	defer func() {
		if e := recover(); e != nil {
			if e, ok := e.(*SyntaxError); ok {
				err = e
				return
			}
			panic(e) // unreachable unless parser has a bug
		}
	}()

	p := &exprParser{s: text}
	x = p.or()
	if p.tok != "" {
		panic(&SyntaxError{Offset: p.pos, Err: "unexpected token " + p.tok})
	}
	return x, nil
}

// or parses a sequence of || expressions.
// On entry, the next input token has not yet been lexed.
// On exit, the next input token has been lexed and is in p.tok.
func (p *exprParser) or() Expr {
	x := p.and()
	for p.tok == "||" {
		x = or(x, p.and())
	}
	return x
}

// and parses a sequence of && expressions.
// On entry, the next input token has not yet been lexed.
// On exit, the next input token has been lexed and is in p.tok.
func (p *exprParser) and() Expr {
	x := p.not()
	for p.tok == "&&" {
		x = and(x, p.not())
	}
	return x
}

// not parses a ! expression.
// On entry, the next input token has not yet been lexed.
// On exit, the next input token has been lexed and is in p.tok.
func (p *exprParser) not() Expr {
	p.size++
	if p.size > maxSize {
		panic(&SyntaxError{Offset: p.pos, Err: "build expression too large"})
	}
	p.lex()
	if p.tok == "!" {
		p.lex()
		if p.tok == "!" {
			panic(&SyntaxError{Offset: p.pos, Err: "double negation not allowed"})
		}
		return not(p.atom())
	}
	return p.atom()
}

// atom parses a tag or a parenthesized expression.
// On entry, the next input token HAS been lexed.
// On exit, the next input token has been lexed and is in p.tok.
func (p *exprParser) atom() Expr {
	// first token already in p.tok
	if p.tok == "(" {
		pos := p.pos
		defer func() {
			if e := recover(); e != nil {
				if e, ok := e.(*SyntaxError); ok && e.Err == "unexpected end of expression" {
					e.Err = "missing close paren"
				}
				panic(e)
			}
		}()
		x := p.or()
		if p.tok != ")" {
			panic(&SyntaxError{Offset: pos, Err: "missing close paren"})
		}
		p.lex()
		return x
	}

	if !p.isTag {
		if p.tok == "" {
			panic(&SyntaxError{Offset: p.pos, Err: "unexpected end of expression"})
		}
		panic(&SyntaxError{Offset: p.pos, Err: "unexpected token " + p.tok})
	}
	tok := p.tok
	p.lex()
	return tag(tok)
}

// lex finds and consumes the next token in the input stream.
// On return, p.tok is set to the token text,
// p.isTag reports whether the token was a tag,
// and p.pos records the byte offset of the start of the token in the input stream.
// If lex reaches the end of the input, p.tok is set to the empty string.
// For any other syntax error, lex panics with a SyntaxError.
func (p *exprParser) lex() {
	p.isTag = false
	for p.i < len(p.s) && (p.s[p.i] == ' ' || p.s[p.i] == '\t') {
		p.i++
	}
	if p.i >= len(p.s) {
		p.tok = ""
		p.pos = p.i
		return
	}
	switch p.s[p.i] {
	case '(', ')', '!':
		p.pos = p.i
		p.i++
		p.tok = p.s[p.pos:p.i]
		return

	case '&', '|':
		if p.i+1 >= len(p.s) || p.s[p.i+1] != p.s[p.i] {
			panic(&SyntaxError{Offset: p.i, Err: "invalid syntax at " + string(rune(p.s[p.i]))})
		}
		p.pos = p.i
		p.i += 2
		p.tok = p.s[p.pos:p.i]
		return
	}

	tag := p.s[p.i:]
	for i, c := range tag {
		if !unicode.IsLetter(c) && !unicode.IsDigit(c) && c != '_' && c != '.' {
			tag = tag[:i]
			break
		}
	}
	if tag == "" {
		c, _ := utf8.DecodeRuneInString(p.s[p.i:])
		panic(&SyntaxError{Offset: p.i, Err: "invalid syntax at " + string(c)})
	}

	p.pos = p.i
	p.i += len(tag)
	p.tok = p.s[p.pos:p.i]
	p.isTag = true
}

// IsPlusBuild reports whether the line of text is a “// +build” constraint.
// It only checks the prefix of the text, not that the expression itself parses.
func IsPlusBuild(line string) bool {
	_, ok := splitPlusBuild(line)
	return ok
}

// splitPlusBuild splits apart the leading // +build prefix in line from the build expression itself.
// It returns "", false if the input is not a // +build line or if the input contains multiple lines.
func splitPlusBuild(line string) (expr string, ok bool) {
	// A single trailing newline is OK; otherwise multiple lines are not.
	if len(line) > 0 && line[len(line)-1] == '\n' {
		line = line[:len(line)-1]
	}
	if strings.Contains(line, "\n") {
		return "", false
	}

	if !strings.HasPrefix(line, "//") {
		return "", false
	}
	line = line[len("//"):]
	// Note the space is optional; "//+build" is recognized too.
	line = strings.TrimSpace(line)

	if !strings.HasPrefix(line, "+build") {
		return "", false
	}
	line = line[len("+build"):]

	// If strings.TrimSpace finds more to trim after removing the +build prefix,
	// it means that the prefix was followed by a space, making this a +build line
	// (as opposed to a +buildsomethingelse line).
	// If line is empty, we had "// +build" by itself, which also counts.
	trim := strings.TrimSpace(line)
	if len(line) == len(trim) && line != "" {
		return "", false
	}

	return trim, true
}

// parsePlusBuildExpr parses a legacy build tag expression (as used with “// +build”).
func parsePlusBuildExpr(text string) (Expr, error) {
	// Only allow up to 100 AND/OR operators for "old" syntax.
	// This is much less than the limit for "new" syntax,
	// but uses of old syntax were always very simple.
	const maxOldSize = 100
	size := 0

	var x Expr
	for _, clause := range strings.Fields(text) {
		var y Expr
		for _, lit := range strings.Split(clause, ",") {
			var z Expr
			var neg bool
			if strings.HasPrefix(lit, "!!") || lit == "!" {
				z = tag("ignore")
			} else {
				if strings.HasPrefix(lit, "!") {
					neg = true
					lit = lit[len("!"):]
				}
				if isValidTag(lit) {
					z = tag(lit)
				} else {
					z = tag("ignore")
				}
				if neg {
					z = not(z)
				}
			}
			if y == nil {
				y = z
			} else {
				if size++; size > maxOldSize {
					return nil, errComplex
				}
				y = and(y, z)
			}
		}
		if x == nil {
			x = y
		} else {
			if size++; size > maxOldSize {
				return nil, errComplex
			}
			x = or(x, y)
		}
	}
	if x == nil {
		x = tag("ignore")
	}
	return x, nil
}

// isValidTag reports whether the word is a valid build tag.
// Tags must be letters, digits, underscores or dots.
// Unlike in Go identifiers, all digits are fine (e.g., "386").
func isValidTag(word string) bool {
	if word == "" {
		return false
	}
	for _, c := range word {
		if !unicode.IsLetter(c) && !unicode.IsDigit(c) && c != '_' && c != '.' {
			return false
		}
	}
	return true
}

var errComplex = errors.New("expression too complex for // +build lines")

// PlusBuildLines returns a sequence of “// +build” lines that evaluate to the build expression x.
// If the expression is too complex to convert directly to “// +build” lines, PlusBuildLines returns an error.
func PlusBuildLines(x Expr) ([]string, error) {
	// Push all NOTs to the expression leaves, so that //go:build !(x && y) can be treated as !x || !y.
	// This rewrite is both efficient and commonly needed, so it's worth doing.
	// Essentially all other possible rewrites are too expensive and too rarely needed.
	x = pushNot(x, false)

	// Split into AND of ORs of ANDs of literals (tag or NOT tag).
	var split [][][]Expr
	for _, or := range appendSplitAnd(nil, x) {
		var ands [][]Expr
		for _, and := range appendSplitOr(nil, or) {
			var lits []Expr
			for _, lit := range appendSplitAnd(nil, and) {
				switch lit.(type) {
				case *TagExpr, *NotExpr:
					lits = append(lits, lit)
				default:
					return nil, errComplex
				}
			}
			ands = append(ands, lits)
		}
		split = append(split, ands)
	}

	// If all the ORs have length 1 (no actual OR'ing going on),
	// push the top-level ANDs to the bottom level, so that we get
	// one // +build line instead of many.
	maxOr := 0
	for _, or := range split {
		if maxOr < len(or) {
			maxOr = len(or)
		}
	}
	if maxOr == 1 {
		var lits []Expr
		for _, or := range split {
			lits = append(lits, or[0]...)
		}
		split = [][][]Expr{{lits}}
	}

	// Prepare the +build lines.
	var lines []string
	for _, or := range split {
		line := "// +build"
		for _, and := range or {
			clause := ""
			for i, lit := range and {
				if i > 0 {
					clause += ","
				}
				clause += lit.String()
			}
			line += " " + clause
		}
		lines = append(lines, line)
	}

	return lines, nil
}

// pushNot applies DeMorgan's law to push negations down the expression,
// so that only tags are negated in the result.
// (It applies the rewrites !(X && Y) => (!X || !Y) and !(X || Y) => (!X && !Y).)
func pushNot(x Expr, not bool) Expr {
	switch x := x.(type) {
	default:
		// unreachable
		return x
	case *NotExpr:
		if _, ok := x.X.(*TagExpr); ok && !not {
			return x
		}
		return pushNot(x.X, !not)
	case *TagExpr:
		if not {
			return &NotExpr{X: x}
		}
		return x
	case *AndExpr:
		x1 := pushNot(x.X, not)
		y1 := pushNot(x.Y, not)
		if not {
			return or(x1, y1)
		}
		if x1 == x.X && y1 == x.Y {
			return x
		}
		return and(x1, y1)
	case *OrExpr:
		x1 := pushNot(x.X, not)
		y1 := pushNot(x.Y, not)
		if not {
			return and(x1, y1)
		}
		if x1 == x.X && y1 == x.Y {
			return x
		}
		return or(x1, y1)
	}
}

// appendSplitAnd appends x to list while splitting apart any top-level && expressions.
// For example, appendSplitAnd({W}, X && Y && Z) = {W, X, Y, Z}.
func appendSplitAnd(list []Expr, x Expr) []Expr {
	if x, ok := x.(*AndExpr); ok {
		list = appendSplitAnd(list, x.X)
		list = appendSplitAnd(list, x.Y)
		return list
	}
	return append(list, x)
}

// appendSplitOr appends x to list while splitting apart any top-level || expressions.
// For example, appendSplitOr({W}, X || Y || Z) = {W, X, Y, Z}.
func appendSplitOr(list []Expr, x Expr) []Expr {
	if x, ok := x.(*OrExpr); ok {
		list = appendSplitOr(list, x.X)
		list = appendSplitOr(list, x.Y)
		return list
	}
	return append(list, x)
}
