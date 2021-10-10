// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package usesgenerics defines an Analyzer that checks for usage of generic
// features added in Go 1.18.
package usesgenerics

import (
	"go/ast"
	"go/types"
	"reflect"
	"strings"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/typeparams"
)

var Analyzer = &analysis.Analyzer{
	Name:       "usesgenerics",
	Doc:        Doc,
	Requires:   []*analysis.Analyzer{inspect.Analyzer},
	Run:        run,
	ResultType: reflect.TypeOf((*Result)(nil)),
	FactTypes:  []analysis.Fact{new(featuresFact)},
}

const Doc = `detect whether a package uses generics features

The usesgenerics analysis reports whether a package directly or transitively
uses certain features associated with generic programming in Go.`

// Result is the usesgenerics analyzer result type. The Direct field records
// features used directly by the package being analyzed (i.e. contained in the
// package source code). The Transitive field records any features used by the
// package or any of its transitive imports.
type Result struct {
	Direct, Transitive Features
}

// Features is a set of flags reporting which features of generic Go code a
// package uses, or 0.
type Features int

const (
	// GenericTypeDecls indicates whether the package declares types with type
	// parameters.
	GenericTypeDecls Features = 1 << iota

	// GenericFuncDecls indicates whether the package declares functions with
	// type parameters.
	GenericFuncDecls

	// EmbeddedTypeSets indicates whether the package declares interfaces that
	// contain structural type restrictions, i.e. are not fully described by
	// their method sets.
	EmbeddedTypeSets

	// TypeInstantiation indicates whether the package instantiates any generic
	// types.
	TypeInstantiation

	// FuncInstantiation indicates whether the package instantiates any generic
	// functions.
	FuncInstantiation
)

func (f Features) String() string {
	var feats []string
	if f&GenericTypeDecls != 0 {
		feats = append(feats, "typeDecl")
	}
	if f&GenericFuncDecls != 0 {
		feats = append(feats, "funcDecl")
	}
	if f&EmbeddedTypeSets != 0 {
		feats = append(feats, "typeSet")
	}
	if f&TypeInstantiation != 0 {
		feats = append(feats, "typeInstance")
	}
	if f&FuncInstantiation != 0 {
		feats = append(feats, "funcInstance")
	}
	return "features{" + strings.Join(feats, ",") + "}"
}

type featuresFact struct {
	Features Features
}

func (f *featuresFact) AFact()         {}
func (f *featuresFact) String() string { return f.Features.String() }

func run(pass *analysis.Pass) (interface{}, error) {
	direct := directFeatures(pass)

	transitive := direct | importedTransitiveFeatures(pass)
	if transitive != 0 {
		pass.ExportPackageFact(&featuresFact{transitive})
	}

	return &Result{
		Direct:     direct,
		Transitive: transitive,
	}, nil
}

// directFeatures computes which generic features are used directly by the
// package being analyzed.
func directFeatures(pass *analysis.Pass) Features {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	nodeFilter := []ast.Node{
		(*ast.FuncType)(nil),
		(*ast.InterfaceType)(nil),
		(*ast.ImportSpec)(nil),
		(*ast.TypeSpec)(nil),
	}

	var direct Features

	inspect.Preorder(nodeFilter, func(node ast.Node) {
		switch n := node.(type) {
		case *ast.FuncType:
			if tparams := typeparams.ForFuncType(n); tparams != nil {
				direct |= GenericFuncDecls
			}
		case *ast.InterfaceType:
			tv := pass.TypesInfo.Types[n]
			if iface, _ := tv.Type.(*types.Interface); iface != nil && !typeparams.IsMethodSet(iface) {
				direct |= EmbeddedTypeSets
			}
		case *ast.TypeSpec:
			if tparams := typeparams.ForTypeSpec(n); tparams != nil {
				direct |= GenericTypeDecls
			}
		}
	})

	instances := typeparams.GetInstances(pass.TypesInfo)
	for _, inst := range instances {
		switch inst.Type.(type) {
		case *types.Named:
			direct |= TypeInstantiation
		case *types.Signature:
			direct |= FuncInstantiation
		}
	}
	return direct
}

// importedTransitiveFeatures computes features that are used transitively via
// imports.
func importedTransitiveFeatures(pass *analysis.Pass) Features {
	var feats Features
	for _, imp := range pass.Pkg.Imports() {
		var importedFact featuresFact
		if pass.ImportPackageFact(imp, &importedFact) {
			feats |= importedFact.Features
		}
	}
	return feats
}
