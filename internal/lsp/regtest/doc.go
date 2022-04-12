// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package regtest provides a framework for writing gopls regression tests.
//
// User reported regressions are often expressed in terms of editor
// interactions. For example: "When I open my editor in this directory,
// navigate to this file, and change this line, I get a diagnostic that doesn't
// make sense". In these cases reproducing, diagnosing, and writing a test to
// protect against this regression can be difficult.
//
// The regtest package provides an API for developers to express these types of
// user interactions in ordinary Go tests, validate them, and run them in a
// variety of execution modes (see gopls/doc/daemon.md for more information on
// execution modes). This is achieved roughly as follows:
//   - the Runner type starts and connects to a gopls instance for each
//     configured execution mode.
//   - the Env type provides a collection of resources to use in writing tests
//     (for example a temporary working directory and fake text editor)
//   - user interactions with these resources are scripted using test wrappers
//     around the API provided by the golang.org/x/tools/internal/lsp/fake
//     package.
//
// Regressions are expressed in terms of Expectations, which at a high level
// are conditions that we expect to be met (or not to be met) at some point
// after performing the interactions in the test. This is necessary because the
// LSP is by construction asynchronous: both client and server can send
// each other notifications without formal acknowledgement that they have been
// fully processed.
//
// Simple Expectations may be combined to match specific conditions reported by
// the user. In the example above, a regtest validating that the user-reported
// bug had been fixed would "expect" that the editor never displays the
// confusing diagnostic.
package regtest
