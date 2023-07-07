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
// to pass through profiling flags to the gopls process. Each of these flags
// sets a suffix for the respective gopls binary profiling flag, which is
// prefixed with a name corresponding to the shared repository or (in some
// cases) benchmark name. For example, setting -gopls_cpuprofile=cpu will
// result in profiles named tools.cpu, iwl.tools.cpu, etc. Here, tools.cpu is
// the cpu profile for the shared x/tools session, which may be used by
// multiple benchmark functions, and iwl.tools.cpu is the cpu profile for the
// last iteration of the initial workspace load test, which starts a new editor
// session for each iteration.
//
// In some cases we want to collect profiles that are bracketed around the
// specific inner operations being tested by the benchmark (for example
// DidChange processing). To support this use-case, recent versions of gopls
// include custom LSP commands to start and stop profiling. If these commands
// are supported, the benchmark runner will instrument certain benchmarks to
// collect profiles and compute a cpu_seconds benchmark metric recording the
// total CPU sampled during the profile. These profile files may be specified
// using benchmark-specific suffix flags, such as -didchange_cpuprofile, in
// which case the profile files will be written according to the naming schema
// described in the previous paragraph, and will not be deleted when the
// benchmark exits. For example, setting -didchange_cpuprofile=change.cpu
// results in a tools.change.cpu file created during BenchmarkDidChange/tools.
//
// TODO(rfindley): simplify these profiling flags to just have a single profile
// per test that "does the right thing".
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
