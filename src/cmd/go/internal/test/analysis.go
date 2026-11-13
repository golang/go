// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

// This file defines go test's use of an analysis driver with a
// dynamically built tool in place of cmd/vet (GOEXPERIMENT=vetfiles;
// see go.dev/issue/82464). For now, its functions are no-op stubs.

import (
	"context"

	"cmd/go/internal/base"
	"cmd/go/internal/load"
	"cmd/go/internal/modload"
	"cmd/go/internal/work"
)

// addTestAnalysis adds the actions to analyze package p, in place of
// cmd/vet (see addTestVet). The test's run action and the install
// action, if any, depend on the analysis.
func addTestAnalysis(ld *modload.Loader, b *work.Builder, p *load.Package, runAction, installAction *work.Action) {
	// TODO(adonovan): implement.
	base.Fatalf("GOEXPERIMENT=vetfiles not yet implemented")
}

// buildAnalysisTool adds the actions to build the analysis tool and
// query its metadata, if any packages need it.
func buildAnalysisTool(ld *modload.Loader, ctx context.Context, b *work.Builder, opts load.PackageOpts) {
	// TODO(adonovan): implement.
	base.Fatalf("GOEXPERIMENT=vetfiles not yet implemented")
}

// runAnalysisToolChild is the main function of a child go test
// command that builds the analysis tool on behalf of its parent.
// It returns true to override the usual behavior of go test.
func runAnalysisToolChild(ctx context.Context, ld *modload.Loader) bool {
	// TODO(adonovan): implement.
	return false
}
