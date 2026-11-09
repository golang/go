// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package contraire

import "fmt"

// newGenerator returns a new generator.
func newGenerator(cfg *config) *generator {
	return &generator{cfg: cfg}
}

// A generator generates Contraire ASTs.
type generator struct {
	cfg *config // immutable
	s   *state  // mutable
}

type state struct {
	constDecls []*constDecl
	typeDecls  []*typeDecl
}

func (g *generator) genProg() *prog {
	g.s = &state{}

	// Generate all of the LHS information for the constant and type declarations.
	// This way, names are available for both forward and backward references
	// when generating the RHS information. This allows for cycles.
	g.s.constDecls = g.declareConstLHSs()
	g.s.typeDecls = g.declareTypeLHSs()

	g.declareConstRHSs()
	g.declareTypeRHSs()

	// Package declarations into a program.
	var decls []decl
	for _, d := range g.s.constDecls {
		decls = append(decls, d)
	}
	for _, d := range g.s.typeDecls {
		decls = append(decls, d)
	}
	return &prog{decls: decls}
}

func (g *generator) declareConstLHSs() []*constDecl {
	decls := make([]*constDecl, oneToN(g.cfg.ConstDeclMaxCount))
	for i := range decls {
		decls[i] = &constDecl{name: fmt.Sprintf("c%d", i)}
	}
	return decls
}

func (g *generator) declareConstRHSs() {
	for _, d := range g.s.constDecls {
		d.expr = g.genConstExpr()
	}
}

func (g *generator) declareTypeLHSs() []*typeDecl {
	decls := make([]*typeDecl, oneToN(g.cfg.TypeDeclMaxCount))
	for i := range decls {
		decls[i] = &typeDecl{name: fmt.Sprintf("T%d", i)}
	}
	return decls
}

func (g *generator) declareTypeRHSs() {
	for _, d := range g.s.typeDecls {
		d.expr = g.genTypeExpr()
	}
}

func (g *generator) genConstExpr() constExpr {
	return chooseCDF(
		alt[constExpr]{g.cfg.PConstName, g.genConstName},
		alt[constExpr]{g.cfg.PConstLtrl, g.genConstLtrl},
	)
}

func (g *generator) genConstName() constExpr {
	return &constName{name: chooseUniform(g.s.constDecls).name}
}

var constLtrlSingleton = &constLtrl{}

func (g *generator) genConstLtrl() constExpr {
	return constLtrlSingleton
}

func (g *generator) genTypeExpr() typeExpr {
	return chooseCDF(
		alt[typeExpr]{g.cfg.PTypeName, g.genTypeDeclName},
		alt[typeExpr]{g.cfg.PTypeLtrl, g.genTypeLtrl},
	)
}

func (g *generator) genTypeDeclName() typeExpr {
	return &typeName{name: chooseUniform(g.s.typeDecls).name}
}

func (g *generator) genTypeLtrl() typeExpr {
	return chooseCDF(
		alt[typeExpr]{g.cfg.PTypeBasic, g.genTypeBasic},
		alt[typeExpr]{g.cfg.PTypePointer, g.genTypePointer},
	)
}

var typeBasicSingleton = &typeBasic{}

func (g *generator) genTypeBasic() typeExpr {
	return typeBasicSingleton
}

func (g *generator) genTypePointer() typeExpr {
	return &typePointer{expr: g.genTypeExpr()}
}
