// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build go1.28

package analysis

import "go/analysis"

// The types of the analysis API have moved to the standard library in
// Go 1.28. The declarations here are now just aliases for the new types.
// Follow the aliases for complete documentation.

type (
	// An Analyzer describes an analysis function and its options.
	Analyzer = analysis.Analyzer

	// A Pass provides information to the Run function that
	// applies a specific analyzer to a single Go package.
	Pass = analysis.Pass

	// PackageFact is a package together with an associated fact.
	PackageFact = analysis.PackageFact

	// ObjectFact is an object together with an associated fact.
	ObjectFact = analysis.ObjectFact

	// The Range interface provides a range.
	// It's equivalent to and satisfied by [ast.Node].
	Range = analysis.Range

	// A Fact is an intermediate fact produced during analysis.
	Fact = analysis.Fact

	// A Module describes the module to which a package belongs.
	Module = analysis.Module

	// ModuleError holds errors loading a module.
	ModuleError = analysis.ModuleError

	// A Diagnostic is a message associated with a source location or range.
	Diagnostic = analysis.Diagnostic

	// RelatedInformation contains information related to a diagnostic.
	RelatedInformation = analysis.RelatedInformation

	// A SuggestedFix is a code change associated with a Diagnostic that a
	// user can choose to apply to their code.
	SuggestedFix = analysis.SuggestedFix

	// A TextEdit represents the replacement of the code between Pos and End with the new text.
	TextEdit = analysis.TextEdit
)
