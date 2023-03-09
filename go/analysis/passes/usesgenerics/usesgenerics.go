// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package usesgenerics

import (
	_ "embed"
	"reflect"

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/analysis/passes/inspect"
	"golang.org/x/tools/go/analysis/passes/internal/analysisutil"
	"golang.org/x/tools/go/ast/inspector"
	"golang.org/x/tools/internal/typeparams/genericfeatures"
)

//go:embed doc.go
var doc string

var Analyzer = &analysis.Analyzer{
	Name:       "usesgenerics",
	Doc:        analysisutil.MustExtractDoc(doc, "usesgenerics"),
	URL:        "https://pkg.go.dev/golang.org/x/tools/go/analysis/passes/usesgenerics",
	Requires:   []*analysis.Analyzer{inspect.Analyzer},
	Run:        run,
	ResultType: reflect.TypeOf((*Result)(nil)),
	FactTypes:  []analysis.Fact{new(featuresFact)},
}

type Features = genericfeatures.Features

const (
	GenericTypeDecls  = genericfeatures.GenericTypeDecls
	GenericFuncDecls  = genericfeatures.GenericFuncDecls
	EmbeddedTypeSets  = genericfeatures.EmbeddedTypeSets
	TypeInstantiation = genericfeatures.TypeInstantiation
	FuncInstantiation = genericfeatures.FuncInstantiation
)

// Result is the usesgenerics analyzer result type. The Direct field records
// features used directly by the package being analyzed (i.e. contained in the
// package source code). The Transitive field records any features used by the
// package or any of its transitive imports.
type Result struct {
	Direct, Transitive Features
}

type featuresFact struct {
	Features Features
}

func (f *featuresFact) AFact()         {}
func (f *featuresFact) String() string { return f.Features.String() }

func run(pass *analysis.Pass) (interface{}, error) {
	inspect := pass.ResultOf[inspect.Analyzer].(*inspector.Inspector)

	direct := genericfeatures.ForPackage(inspect, pass.TypesInfo)

	transitive := direct | importedTransitiveFeatures(pass)
	if transitive != 0 {
		pass.ExportPackageFact(&featuresFact{transitive})
	}

	return &Result{
		Direct:     direct,
		Transitive: transitive,
	}, nil
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
