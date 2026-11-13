// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package work

// This file defines the build actions for go test's use of an
// analysis driver in place of cmd/vet (GOEXPERIMENT=vetfiles).
// For now, it contains only no-op stubs.

// analysisAction holds the state of an analysisvet action.
type analysisAction struct {
	// TODO(adonovan): implement.
}

// analysisState is the per-Builder state of analysis in go test.
type analysisState struct {
	// TODO(adonovan): implement.
}

// prioritizeAnalysisTool gives the actions that build the analysis
// tool and query its metadata precedence over all others.
func (b *Builder) prioritizeAnalysisTool(all []*Action) {
	// TODO(adonovan): implement.
}
