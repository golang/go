// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package gotestchecker

// This file defines the hooks by which the analysis of one package
// records its costs. In this version they do nothing.
//
// TODO(adonovan): implement profiling (go test -debug-vetprofile),
// in pending CL 856982.

import (
	"go/analysis"
	"internal/analysis/gotestchecker/protocol"
)

// Phases of the analysis of one package.
const (
	phaseParse     = "parse"     // parsing Go files
	phaseTypecheck = "typecheck" // type checking (including reading export data of imports)
	phaseExport    = "export"    // writing export data
	phaseFacts     = "facts"     // decoding and encoding an analyzer's facts
	phaseRun       = "run"       // an analyzer's Run function
	phaseWrite     = "write"     // writing pass files
)

// A profiler records the costs of one tool process.
// A nil *profiler is valid and does nothing.
type profiler struct{}

// startProfiler starts profiling if the request asks for it,
// and returns nil otherwise.
func startProfiler(areq *protocol.AnalyzeRequest) (*profiler, error) { return nil, nil }

// setAnalyzers records the analyzers of the tool, in index order.
func (p *profiler) setAnalyzers(analyzers []*analysis.Analyzer) {}

// do calls f, attributing its costs to the specified phase and
// analyzer (if not -1).
func (p *profiler) do(phase string, analyzer int, f func()) { f() }

// doErr is like do, but returns the error result of f.
func (p *profiler) doErr(phase string, analyzer int, f func() error) error { return f() }

// factsOut records the facts encoded by an analyzer.
func (p *profiler) factsOut(analyzer int, own, deps, bytes int) {}

// factsIn records the size of the facts of a direct import decoded
// for an analyzer.
func (p *profiler) factsIn(analyzer int, bytes int) {}

// factPassFile records the reading of a pass file of a direct import
// for an analyzer, and whether it was empty (no facts or failure).
func (p *profiler) factPassFile(analyzer int, empty bool) {}

// factsDecoded records the number of facts decoded for an analyzer,
// and the number of those discarded.
func (p *profiler) factsDecoded(analyzer int, decoded, discarded int) {}

// finish stops profiling, writes the profile, and writes the stats
// file, attributing CPU time to labels.
func (p *profiler) finish() error { return nil }
