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
// variety of execution modes.
//
// # Test package setup
//
// The regression test package uses a couple of uncommon patterns to reduce
// boilerplate in test bodies. First, it is intended to be imported as "." so
// that helpers do not need to be qualified. Second, it requires some setup
// that is currently implemented in the regtest.Main function, which must be
// invoked by TestMain. Therefore, a minimal regtest testing package looks
// like this:
//
//	package lsptests
//
//	import (
//		"fmt"
//		"testing"
//
//		"golang.org/x/tools/gopls/internal/hooks"
//		. "golang.org/x/tools/gopls/internal/lsp/regtest"
//	)
//
//	func TestMain(m *testing.M) {
//		Main(m, hooks.Options)
//	}
//
// # Writing a simple regression test
//
// To run a regression test use the regtest.Run function, which accepts a
// txtar-encoded archive defining the initial workspace state. This function
// sets up the workspace in a temporary directory, creates a fake text editor,
// starts gopls, and initializes an LSP session. It then invokes the provided
// test function with an *Env handle encapsulating the newly created
// environment. Because gopls may be run in various modes (as a sidecar or
// daemon process, with different settings), the test runner may perform this
// process multiple times, re-running the test function each time with a new
// environment.
//
//	func TestOpenFile(t *testing.T) {
//		const files = `
//	-- go.mod --
//	module mod.com
//
//	go 1.12
//	-- foo.go --
//	package foo
//	`
//		Run(t, files, func(t *testing.T, env *Env) {
//			env.OpenFile("foo.go")
//		})
//	}
//
// # Configuring Regtest Execution
//
// The regtest package exposes several options that affect the setup process
// described above. To use these options, use the WithOptions function:
//
//	WithOptions(opts...).Run(...)
//
// See options.go for a full list of available options.
//
// # Operating on editor state
//
// To operate on editor state within the test body, the Env type provides
// access to the workspace directory (Env.SandBox), text editor (Env.Editor),
// LSP server (Env.Server), and 'awaiter' (Env.Awaiter).
//
// In most cases, operations on these primitive building blocks of the
// regression test environment expect a Context (which should be a child of
// env.Ctx), and return an error. To avoid boilerplate, the Env exposes a set
// of wrappers in wrappers.go for use in scripting:
//
//	env.CreateBuffer("c/c.go", "")
//	env.EditBuffer("c/c.go", fake.Edit{
//		Text: `package c`,
//	})
//
// These wrappers thread through Env.Ctx, and call t.Fatal on any errors.
//
// # Expressing expectations
//
// The general pattern for a regression test is to script interactions with the
// fake editor and sandbox, and assert that gopls behaves correctly after each
// state change. Unfortunately, this is complicated by the fact that state
// changes are communicated to gopls via unidirectional client->server
// notifications (didOpen, didChange, etc.), and resulting gopls behavior such
// as diagnostics, logs, or messages is communicated back via server->client
// notifications. Therefore, within regression tests we must be able to say "do
// this, and then eventually gopls should do that". To achieve this, the
// regtest package provides a framework for expressing conditions that must
// eventually be met, in terms of the Expectation type.
//
// To express the assertion that "eventually gopls must meet these
// expectations", use env.Await(...):
//
//	env.RegexpReplace("x/x.go", `package x`, `package main`)
//	env.Await(env.DiagnosticAtRegexp("x/main.go", `fmt`))
//
// Await evaluates the provided expectations atomically, whenever the client
// receives a state-changing notification from gopls. See expectation.go for a
// full list of available expectations.
//
// A fundamental problem with this model is that if gopls never meets the
// provided expectations, the test runner will hang until the test timeout
// (which defaults to 10m). There are two ways to work around this poor
// behavior:
//
//  1. Use a precondition to define precisely when we expect conditions to be
//     met. Gopls provides the OnceMet(precondition, expectations...) pattern
//     to express ("once this precondition is met, the following expectations
//     must all hold"). To instrument preconditions, gopls uses verbose
//     progress notifications to inform the client about ongoing work (see
//     CompletedWork). The most common precondition is to wait for gopls to be
//     done processing all change notifications, for which the regtest package
//     provides the AfterChange helper. For example:
//
//     // We expect diagnostics to be cleared after gopls is done processing the
//     // didSave notification.
//     env.SaveBuffer("a/go.mod")
//     env.AfterChange(EmptyDiagnostics("a/go.mod"))
//
//  2. Set a shorter timeout during development, if you expect to be breaking
//     tests. By setting the environment variable GOPLS_REGTEST_TIMEOUT=5s,
//     regression tests will time out after 5 seconds.
//
// # Tips & Tricks
//
// Here are some tips and tricks for working with regression tests:
//
//  1. Set the environment variable GOPLS_REGTEST_TIMEOUT=5s during development.
//  2. Run tests with  -short. This will only run regression tests in the
//     default gopls execution mode.
//  3. Use capture groups to narrow regexp positions. All regular-expression
//     based positions (such as DiagnosticAtRegexp) will match the position of
//     the first capture group, if any are provided. This can be used to
//     identify a specific position in the code for a pattern that may occur in
//     multiple places. For example `var (mu) sync.Mutex` matches the position
//     of "mu" within the variable declaration.
//  4. Read diagnostics into a variable to implement more complicated
//     assertions about diagnostic state in the editor. To do this, use the
//     pattern OnceMet(precondition, ReadDiagnostics("file.go", &d)) to capture
//     the current diagnostics as soon as the precondition is met. This is
//     preferable to accessing the diagnostics directly, as it avoids races.
package regtest
