// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package scripttest adapts the script engine for use in tests.
package scripttest

// This package provides APIs for executing "script" tests; this
// way of writing Go tests originated with the Go command, and has
// since been generalized to work with other commands, such as the
// compiler, linker, and other tools.
//
// The top level entry point for this package is "Test", which
// accepts a previously configured script engine and pattern (typically
// by convention this will be "testdata/script/*.txt")
// then kicks off the engine on each file that matches the
// pattern.
