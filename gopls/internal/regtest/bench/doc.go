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
// Benchmark functions run gopls in a separate process, which means the normal
// test flags for profiling aren't useful. Instead the -gopls_cpuprofile,
// -gopls_memprofile, -gopls_allocprofile, and -gopls_trace flags may be used
// to pass through profiling to the gopls subproces.
//
// Each of these flags sets a suffix for the respective gopls profile, which is
// named according to the schema <repo>.<operation>.<suffix>. For example,
// setting -gopls_cpuprofile=cpu will result in profiles named tools.iwl.cpu,
// tools.rename.cpu, etc. In some cases, these profiles are for the entire
// gopls subprocess (as in the initial workspace load), whereas in others they
// span only the critical section of the benchmark. It is up to each benchmark
// to implement profiling as appropriate.
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
