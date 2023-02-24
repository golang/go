// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package tests defines an Analyzer that checks for common mistaken
// usages of tests and examples.
//
// # Analyzer tests
//
// tests: check for common mistaken usages of tests and examples
//
// The tests checker walks Test, Benchmark, Fuzzing and Example functions checking
// malformed names, wrong signatures and examples documenting non-existent
// identifiers.
//
// Please see the documentation for package testing in golang.org/pkg/testing
// for the conventions that are enforced for Tests, Benchmarks, and Examples.
package tests
