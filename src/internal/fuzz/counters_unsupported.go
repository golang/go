// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO: expand the set of supported platforms, with testing. Nothing about
// the instrumentation is OS specific, but only amd64 and arm64 are
// supported in the runtime. See src/runtime/libfuzzer*.
//
// If you update this constraint, also update internal/platform.FuzzInstrumented.
//
//go:build !((darwin || linux || windows || freebsd || openbsd) && (amd64 || arm64))

package fuzz

// TODO(#48504): re-enable on platforms where instrumentation works.
// In theory, we shouldn't need this file at all: if the binary was built
// without coverage, then _counters and _ecounters should have the same address.
// However, this caused an init failure on aix/ppc64, so it's disabled here.

// coverage returns a []byte containing unique 8-bit counters for each edge of
// the instrumented source code. This coverage data will only be generated if
// `-d=libfuzzer` is set at build time. This can be used to understand the code
// coverage of a test execution.
func coverage() []byte { return nil }
