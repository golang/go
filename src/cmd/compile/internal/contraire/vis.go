// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package contraire

import "strings"

// scorer is a visitor that computes a complexity score of an AST.
type scorer struct{ acc int }

func (s *scorer) score(n node) int {
	s.visit(n)
	res := s.acc
	s.acc = 0
	return res
}

func (s *scorer) visit(n node) { n.receive(s) }
func (s *scorer) visitProg(x *prog) {
	for _, d := range x.decls {
		s.visit(d)
	}
}
func (s *scorer) visitConstDecl(x *constDecl)     { s.visit(x.expr) }
func (s *scorer) visitTypeDecl(x *typeDecl)       { s.visit(x.expr) }
func (s *scorer) visitConstName(x *constName)     { s.acc++ }
func (s *scorer) visitConstLtrl(x *constLtrl)     { s.acc++ }
func (s *scorer) visitTypeName(x *typeName)       { s.acc++ }
func (s *scorer) visitTypeBasic(x *typeBasic)     { s.acc++ }
func (s *scorer) visitTypePointer(x *typePointer) { s.acc++; s.visit(x.expr) }

// printer is a visitor that produces a string representation of an AST.
type printer struct{ acc strings.Builder }

func (p *printer) print(n node) string {
	p.visit(n)
	res := p.acc.String()
	p.acc.Reset()
	return res
}

func (p *printer) visit(n node) { n.receive(p) }
func (p *printer) visitProg(x *prog) {
	p.acc.WriteString("package p\n\n")
	for _, d := range x.decls {
		p.visit(d)
		p.acc.WriteString("\n")
	}
}
func (p *printer) visitConstDecl(x *constDecl) {
	p.acc.WriteString("const ")
	p.acc.WriteString(x.name)
	p.acc.WriteString(" = ")
	p.visit(x.expr)
}
func (p *printer) visitTypeDecl(x *typeDecl) {
	p.acc.WriteString("type ")
	p.acc.WriteString(x.name)
	p.acc.WriteString(" ")
	p.visit(x.expr)
}
func (p *printer) visitConstName(x *constName)     { p.acc.WriteString(x.name) }
func (p *printer) visitConstLtrl(x *constLtrl)     { p.acc.WriteString("1") }
func (p *printer) visitTypeName(x *typeName)       { p.acc.WriteString(x.name) }
func (p *printer) visitTypeBasic(x *typeBasic)     { p.acc.WriteString("int") }
func (p *printer) visitTypePointer(x *typePointer) { p.acc.WriteString("*"); p.visit(x.expr) }
