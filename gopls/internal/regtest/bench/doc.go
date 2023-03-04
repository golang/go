// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// The bench package implements benchmarks for various LSP operations.
//
// Benchmarks check out specific commits of popular and/or exemplary
// repositories, and script an external gopls process via a fake text editor.
// By default, benchmarks run the test executable as gopls (using a special
// "gopls mode" environment variable). A different gopls binary may be used by
// setting the -gopls_path or -gopls_commit flags.
//
// This package is a work in progress.
//
// # Profiling
//
// As benchmark functions run gopls in a separate process, the normal test
// flags for profiling are not useful. Instead the -gopls_cpuprofile,
// -gopls_memprofile, -gopls_allocprofile, and -gopls_trace flags may be used
// to pass through profiling flags to the gopls process. Each of these flags
// sets a suffix for the respective gopls profiling flag, which is prefixed
// with a name corresponding to the shared repository or (in some cases)
// benchmark name. For example, settings -gopls_cpuprofile=cpu.out will result
// in profiles named tools.cpu.out, BenchmarkInitialWorkspaceLoad.cpu.out, etc.
// Here, tools.cpu.out is the cpu profile for the shared x/tools session, which
// may be used by multiple benchmark functions, and
// BenchmarkInitialWorkspaceLoad is the cpu profile for the last iteration of
// the initial workspace load test, which starts a new editor session for each
// iteration.
//
// # Integration with perf.golang.org
//
// Benchmarks that run with -short are automatically tracked by
// perf.golang.org, at
// https://perf.golang.org/dashboard/?benchmark=all&repository=tools&branch=release-branch.go1.20
//
// # TODO
//   - add more benchmarks, and more repositories
//   - fix the perf dashboard to not require the branch= parameter
//   - improve this documentation
package bench
