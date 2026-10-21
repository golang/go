// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"go/scanner"
	"go/token"
	"os"
	"strings"
)

// logosCgoSyntaxMsg is the diagnostic for loGo/s syntax in a file that
// imports "C" (SPEC T-304). cmd/cgo parses its inputs with go/parser, so
// such a file never reaches the compiler.
const logosCgoSyntaxMsg = `[correctness::cgo_file_logos_syntax] files that import "C" must use Go syntax only`

// logosSyntaxPos returns the position of the first loGo/s spelling in
// src: the character ? (try, option, and coalescing operators), a $
// glued to a string (interpolation), the adjacent tokens => (lambdas and
// switch arms) and .. (spread), a . that starts a line after an
// automatic semicolon (leading-dot chains), and the plain-Go-token
// families errdefer, yield, scope defer, go group (and go child
// blocks), const if/for/switch/func, and tuple commas and colons. It
// reports false unless src also imports "C", so the diagnostic names a
// cgo file only.
//
// The scan reads Go tokens, so comments, strings, and runes never count,
// and an ordinary Go syntax error leaves no such spelling. The
// plain-token families mirror the compiler parser recognizers, which
// cmd/cgo cannot import (cmd/compile/internal is internal to the
// compiler): errdefer takes a call (an identifier or func follows),
// yield takes an operand (a literal, identifier, or func/struct/map/
// chan/interface follows), scope defer pairs the two words, go group
// needs `(` after group (and `{` after the argument list, on the same
// line, since a newline ends the go statement in both languages), go
// takes a bare block, const takes if/for/switch/func, and a comma or
// colon at the top of a non-call paren is a tuple (call parens follow a
// name, a closer, or func; a name after var, const, type, or inside a
// signature paren cannot start a call). Multiword families report the
// family start (scope, go, const), matching the compiler statement
// positions. Every other use of those words keeps its Go meaning, so
// broken Go without a loGo/s spelling keeps the parser error.
func logosSyntaxPos(name string, src []byte) (token.Position, bool) {
	fs := token.NewFileSet()
	file := fs.AddFile(name, fs.Base(), len(src))
	var s scanner.Scanner
	s.Init(file, src, nil, scanner.ScanComments)

	var (
		found      token.Position
		sawLogos   bool
		sawC       bool
		prev       token.Token
		prevLit    string
		prevOffset int
		prevPrev   token.Token
	)
	note := func(pos token.Pos) {
		if !sawLogos {
			found = fs.Position(pos)
			sawLogos = true
		}
	}
	isLiteral := func(t token.Token) bool {
		switch t {
		case token.INT, token.FLOAT, token.IMAG, token.CHAR, token.STRING:
			return true
		}
		return false
	}
	isOperandStart := func(t token.Token) bool {
		if t == token.IDENT {
			return true
		}
		if isLiteral(t) {
			return true
		}
		switch t {
		case token.FUNC, token.STRUCT, token.MAP, token.CHAN, token.INTERFACE:
			return true
		}
		return false
	}

	// pendName tracks an identifier that may start an identifier-led
	// loGo/s statement; the next token decides. pendNone clears it.
	const (
		pendNone = iota
		pendErrdefer
		pendYield
	)
	pend := pendNone
	var pendPos token.Pos
	// groupArmed tracks `go group` awaiting `(`; braceWait tracks a
	// closed `go group (...)` awaiting `{` on the same line. goPos is
	// the family start, matching the compiler GroupStmt position.
	groupArmed := false
	var goPos token.Pos
	braceWait := false
	braceLine := 0
	// stack tracks open brackets for tuple detection: a comma or colon
	// at the top of a paren whose commas cannot be separators is a
	// tuple. Calls, signatures, and closers keep the Go meaning; a name
	// in declaration position (var t (...), const t (...)) or a name
	// inside a signature paren cannot start a call, so its paren is a
	// tuple.
	type opener struct {
		tok     token.Token
		params  bool
		commaOK bool
		group   bool
	}
	var stack []opener

	for {
		pos, tok, lit := s.Scan()
		if tok == token.EOF {
			break
		}
		if tok == token.COMMENT {
			continue
		}
		if pend != pendNone {
			switch {
			case pend == pendErrdefer && (tok == token.IDENT || tok == token.FUNC):
				note(pendPos)
			case pend == pendYield && isOperandStart(tok):
				note(pendPos)
			}
			pend = pendNone
		}
		if groupArmed {
			groupArmed = false
			if tok == token.LPAREN {
				stack = append(stack, opener{tok: token.LPAREN, commaOK: true, group: true})
				prevPrev = prev
				prev, prevLit, prevOffset = tok, lit, file.Offset(pos)
				continue
			}
		}
		if braceWait {
			braceWait = false
			if tok == token.LBRACE && fs.Position(pos).Line == braceLine {
				note(goPos)
			}
		}
		offset := file.Offset(pos)
		switch {
		case tok == token.ILLEGAL && lit == "?":
			note(pos)
		case tok == token.STRING && prev == token.ILLEGAL && prevLit == "$" && offset == prevOffset+1:
			note(file.Pos(prevOffset))
		case tok == token.GTR && prev == token.ASSIGN && offset == prevOffset+1:
			note(file.Pos(prevOffset))
		case tok == token.PERIOD && prev == token.PERIOD && offset == prevOffset+1:
			note(file.Pos(prevOffset))
		case tok == token.PERIOD && prev == token.SEMICOLON && prevLit == "\n":
			note(pos)
		case tok == token.IDENT && lit == "errdefer" && prev != token.PERIOD:
			pend, pendPos = pendErrdefer, pos
		case tok == token.IDENT && lit == "yield" && prev != token.PERIOD:
			pend, pendPos = pendYield, pos
		case tok == token.DEFER && prev == token.IDENT && prevLit == "scope":
			note(file.Pos(prevOffset))
		case tok == token.IDENT && lit == "group" && prev == token.GO:
			groupArmed, goPos = true, file.Pos(prevOffset)
		case tok == token.LBRACE && prev == token.GO:
			note(file.Pos(prevOffset))
		case tok == token.IF && prev == token.CONST ||
			tok == token.FOR && prev == token.CONST ||
			tok == token.SWITCH && prev == token.CONST ||
			tok == token.FUNC && prev == token.CONST:
			note(file.Pos(prevOffset))
		case tok == token.LPAREN:
			params := prev == token.FUNC || (prev == token.IDENT && prevPrev == token.FUNC)
			enclParams := len(stack) > 0 && stack[len(stack)-1].params
			callLike := prev == token.FUNC || prev == token.RPAREN ||
				prev == token.RBRACK || prev == token.RBRACE ||
				(prev == token.IDENT && prevPrev != token.VAR && prevPrev != token.CONST && prevPrev != token.TYPE && !enclParams)
			stack = append(stack, opener{tok: tok, params: params, commaOK: params || callLike})
		case tok == token.LBRACK || tok == token.LBRACE:
			stack = append(stack, opener{tok: tok, commaOK: true})
		case tok == token.RPAREN || tok == token.RBRACK || tok == token.RBRACE:
			if n := len(stack); n > 0 {
				want := token.LPAREN
				if tok == token.RBRACK {
					want = token.LBRACK
				} else if tok == token.RBRACE {
					want = token.LBRACE
				}
				if stack[n-1].tok == want {
					if stack[n-1].group {
						braceWait = true
						braceLine = fs.Position(pos).Line
					}
					stack = stack[:n-1]
				}
			}
		case tok == token.COMMA || tok == token.COLON:
			if n := len(stack); n > 0 && stack[n-1].tok == token.LPAREN && !stack[n-1].commaOK {
				note(pos)
			}
		case tok == token.STRING && lit == `"C"` && (prev == token.IMPORT || prev == token.LPAREN || prev == token.SEMICOLON):
			sawC = true
		}
		prevPrev = prev
		prev, prevLit, prevOffset = tok, lit, offset
	}
	return found, sawLogos && sawC
}

// logosRejectSyntax runs when go/parser has rejected a cgo input. For a
// loGo/s input that holds loGo/s syntax it prints the pinned diagnostic in
// place of the parser's follow-on errors and exits like any cgo error. It
// returns, leaving the parser errors to print, for every other failure.
func logosRejectSyntax(name string, src []byte) {
	if !isLogosCgoInput(name) {
		return
	}
	pos, ok := logosSyntaxPos(name, src)
	if !ok {
		return
	}
	fmt.Fprintf(os.Stderr, "%s: %s\n", pos, logosCgoSyntaxMsg)
	os.Exit(2)
}

// isLogosCgoInput reports whether name is a loGo/s source file accepted as
// cgo input alongside .go files (M0-S5 routing: cmd/go passes logos files
// that import "C" to cmd/cgo under their .lgo name). It matches ".lgo" and
// the three alias extensions after stripping trailing U+FE0E and U+FE0F
// variation selectors, the same rule as cfg.IsLogosFile. Accepted files
// are parsed with go/parser, so only Go-syntax content compiles here;
// loGo/s syntax is rejected by the parser and reported as
// [correctness::cgo_file_logos_syntax] by logosRejectSyntax (M20-S0).
// Generated outputs keep the input stem (for example "main.lgo.cgo1.go"),
// which satisfies the M0-S3 compiler exemption for names ending in
// .cgo1.go.
func isLogosCgoInput(name string) bool {
	trimmed := strings.TrimRight(name, "\uFE0E\uFE0F")
	if strings.HasSuffix(trimmed, ".lgo") {
		return true
	}
	if strings.HasSuffix(trimmed, ".📜") {
		return true
	}
	if strings.HasSuffix(trimmed, ".⚡") {
		return true
	}
	if strings.HasSuffix(trimmed, ".ʕ◔ϖ◔ʔ") {
		return true
	}
	return false
}
