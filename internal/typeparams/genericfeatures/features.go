// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The genericfeatures package provides utilities for detecting usage of
// generic programming in Go packages.
package genericfeatures

import (
	"go/ast"
	"go/types"
	"strings"

	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/typeparams"
)

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

// ForPackage computes which generic features are used directly by the
// package being analyzed.
func ForPackage(inspect *inspector.Inspector, info *types.Info) Features {
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
			tv := info.Types[n]
			if iface, _ := tv.Type.(*types.Interface); iface != nil && !typeparams.IsMethodSet(iface) {
				direct |= EmbeddedTypeSets
			}
		case *ast.TypeSpec:
			if tparams := typeparams.ForTypeSpec(n); tparams != nil {
				direct |= GenericTypeDecls
			}
		}
	})

	instances := typeparams.GetInstances(info)
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
