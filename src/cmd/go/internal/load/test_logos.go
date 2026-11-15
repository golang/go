// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package load

import (
	"go/ast"
	"go/parser"
	"go/scanner"
	"go/token"
	"internal/logosfile"
	"io"

	"cmd/go/internal/cfg"
)

func parseTestFile(fset *token.FileSet, filename string, src io.Reader) (*ast.File, error) {
	if !logosfile.Is(filename) {
		return parser.ParseFile(fset, filename, src, parser.ParseComments|parser.SkipObjectResolution)
	}
	data, err := io.ReadAll(src)
	if err != nil {
		return nil, err
	}
	return scanLogosTestFile(fset, filename, data), nil
}

type logosScan struct {
	file     *token.File
	scan     scanner.Scanner
	pos      token.Pos
	tok      token.Token
	lit      string
	out      *ast.File
	lastLine int
	inGroup  bool
}

func scanLogosTestFile(fset *token.FileSet, filename string, data []byte) *ast.File {
	file := fset.AddFile(filename, -1, len(data))
	s := &logosScan{file: file, out: &ast.File{Name: &ast.Ident{Name: "p"}}}
	s.scan.Init(file, data, nil, scanner.ScanComments)
	s.next()
	depth := 0
	for s.tok != token.EOF {
		switch s.tok {
		case token.LBRACE:
			depth++
		case token.RBRACE:
			depth--
		case token.FUNC:
			if depth == 0 {
				s.funcDecl()
				continue
			}
		}
		s.next()
	}
	return s.out
}

func (s *logosScan) next() {
	for {
		s.pos, s.tok, s.lit = s.scan.Scan()
		if s.tok != token.COMMENT {
			s.inGroup = false
			return
		}
		s.addComment()
	}
}

func (s *logosScan) addComment() {
	c := &ast.Comment{Slash: s.pos, Text: s.lit}
	line := s.file.Line(s.pos)
	groups := s.out.Comments
	if s.inGroup && len(groups) > 0 && line <= s.lastLine+1 {
		g := groups[len(groups)-1]
		g.List = append(g.List, c)
	} else {
		s.out.Comments = append(groups, &ast.CommentGroup{List: []*ast.Comment{c}})
	}
	s.inGroup = true
	s.lastLine = s.file.Line(token.Pos(int(s.pos) + len(s.lit) - 1))
}

func (s *logosScan) funcDecl() {
	funcPos := s.pos
	s.next()
	if s.tok != token.IDENT {
		return
	}
	decl := &ast.FuncDecl{Name: &ast.Ident{NamePos: s.pos, Name: s.lit}}
	decl.Type = &ast.FuncType{Func: funcPos, Params: &ast.FieldList{}}
	s.next()
	if s.tok == token.LBRACK {
		decl.Type.TypeParams = &ast.FieldList{List: []*ast.Field{{}}}
		s.skipGroup(token.LBRACK, token.RBRACK)
	}
	if s.tok != token.LPAREN {
		return
	}
	decl.Type.Params.List = s.params()
	decl.Type.Results = s.results()
	if s.tok == token.LBRACE {
		decl.Body = s.body()
	}
	s.out.Decls = append(s.out.Decls, decl)
}

func (s *logosScan) skipGroup(open, closing token.Token) {
	depth := 0
	for s.tok != token.EOF {
		switch s.tok {
		case open:
			depth++
		case closing:
			depth--
		}
		s.next()
		if depth == 0 {
			return
		}
	}
}

func (s *logosScan) params() []*ast.Field {
	var toks []token.Token
	var lits []string
	depth := 0
	for s.tok != token.EOF {
		switch s.tok {
		case token.LPAREN:
			depth++
		case token.RPAREN:
			depth--
		}
		if depth > 0 && !(depth == 1 && s.tok == token.LPAREN) {
			toks = append(toks, s.tok)
			lits = append(lits, s.lit)
		}
		s.next()
		if depth == 0 {
			break
		}
	}
	return paramFields(toks, lits)
}

func paramFields(toks []token.Token, lits []string) []*ast.Field {
	if len(toks) == 0 {
		return nil
	}
	if toks[len(toks)-1] == token.COMMA {
		toks = toks[:len(toks)-1]
		lits = lits[:len(lits)-1]
	}
	var names []*ast.Ident
	if len(toks) > 1 && toks[0] == token.IDENT && toks[1] == token.MUL {
		names = []*ast.Ident{{Name: lits[0]}}
		toks, lits = toks[1:], lits[1:]
	}
	if len(toks) >= 2 && toks[0] == token.MUL && toks[1] == token.IDENT {
		switch {
		case len(toks) == 2:
			return []*ast.Field{{Names: names, Type: &ast.StarExpr{X: &ast.Ident{Name: lits[1]}}}}
		case len(toks) == 4 && toks[2] == token.PERIOD && toks[3] == token.IDENT:
			sel := &ast.SelectorExpr{X: &ast.Ident{Name: lits[1]}, Sel: &ast.Ident{Name: lits[3]}}
			return []*ast.Field{{Names: names, Type: &ast.StarExpr{X: sel}}}
		}
	}
	return []*ast.Field{{Type: &ast.BadExpr{}}}
}

func (s *logosScan) results() *ast.FieldList {
	var res *ast.FieldList
	depth := 0
	for s.tok != token.EOF {
		switch s.tok {
		case token.LPAREN, token.LBRACK:
			depth++
		case token.RPAREN, token.RBRACK:
			depth--
		case token.LBRACE:
			if depth == 0 {
				return res
			}
		case token.SEMICOLON:
			if depth == 0 {
				return res
			}
		}
		res = &ast.FieldList{List: []*ast.Field{{}}}
		s.next()
	}
	return res
}

func (s *logosScan) body() *ast.BlockStmt {
	b := &ast.BlockStmt{Lbrace: s.pos}
	depth := 0
	for s.tok != token.EOF {
		switch s.tok {
		case token.LBRACE:
			depth++
		case token.RBRACE:
			depth--
		}
		if depth == 0 {
			b.Rbrace = s.pos
			s.next()
			return b
		}
		s.next()
	}
	b.Rbrace = s.pos
	return b
}

func logosTestXTest(p, pxtest *Package) {
	if !cfg.Logos || pxtest == nil || !p.Standard {
		return
	}
	pxtest.Standard = true
}
