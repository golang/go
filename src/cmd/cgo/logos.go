// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"fmt"
	"go/scanner"
	"go/token"
	"internal/logosfile"
	"os"
)

const logosCgoSyntaxMsg = `[correctness::cgo_file_logos_syntax] files that import "C" must use Go syntax only`

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

	const (
		pendNone = iota
		pendErrdefer
		pendYield
	)
	pend := pendNone
	var pendPos token.Pos
	groupArmed := false
	var goPos token.Pos
	braceWait := false
	braceLine := 0
	type opener struct {
		tok     token.Token
		params  bool
		commaOK bool
		group   bool
		iface   bool
		strct   bool
	}
	var stack []opener
	top := func() opener {
		if n := len(stack); n > 0 {
			return stack[n-1]
		}
		return opener{}
	}
	startsStatement := func(t token.Token) bool {
		switch t {
		case token.ILLEGAL, token.SEMICOLON, token.LBRACE, token.COLON:
			return true
		}
		return false
	}

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
		case tok == token.COLON && prev == token.STRUCT:
			note(pos)
		case tok == token.LBRACK && prev == token.PERIOD && offset == prevOffset+1:
			note(file.Pos(prevOffset))
		case tok == token.CONST && (!startsStatement(prev) || top().iface):
			note(pos)
		case tok == token.ASSIGN && top().strct:
			note(pos)
		case tok == token.LPAREN:
			params := prev == token.FUNC || (prev == token.IDENT && prevPrev == token.FUNC)
			enclParams := len(stack) > 0 && stack[len(stack)-1].params
			callLike := prev == token.FUNC || prev == token.RPAREN ||
				prev == token.RBRACK || prev == token.RBRACE ||
				(prev == token.IDENT && prevPrev != token.VAR && prevPrev != token.CONST && prevPrev != token.TYPE && !enclParams)
			stack = append(stack, opener{tok: tok, params: params, commaOK: params || callLike})
		case tok == token.LBRACK || tok == token.LBRACE:
			stack = append(stack, opener{
				tok:     tok,
				commaOK: true,
				iface:   tok == token.LBRACE && prev == token.INTERFACE,
				strct:   tok == token.LBRACE && prev == token.STRUCT,
			})
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

func logosRejectSyntax(name string, src []byte) {
	if !logosfile.Is(name) {
		return
	}
	pos, ok := logosSyntaxPos(name, src)
	if !ok {
		return
	}
	fmt.Fprintf(os.Stderr, "%s: %s\n", pos, logosCgoSyntaxMsg)
	os.Exit(2)
}
