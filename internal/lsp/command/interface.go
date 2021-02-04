// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Package command defines the interface provided by gopls for the
// workspace/executeCommand LSP request.
//
// This interface is fully specified by the Interface type, provided it
// conforms to the restrictions outlined in its doc string.
//
// Bindings for server-side command dispatch and client-side serialization are
// also provided by this package, via code generation.
package command

//go:generate go run generate.go

import "golang.org/x/tools/internal/lsp/protocol"

// Interface defines the interface gopls exposes for the
// workspace/executeCommand request.
//
// This interface is used to generate marshaling/unmarshaling code, dispatch,
// and documentation, and so has some additional restrictions:
//  1. All method arguments must be JSON serializable.
//  2. Methods must return either error or (T, error), where T is a
//     JSON serializable type.
//  3. The first line of the doc string is special. Everything after the colon
//     is considered the command 'Title'.
//     TODO(rFindley): reconsider this -- Title may be unnecessary.
type Interface interface {
	// RunTests: Run test(s)
	//
	// Runs `go test` for a specific set of test or benchmark functions.
	RunTests(RunTestsArgs) error

	// Generate: Run go generate
	//
	// Runs `go generate` for a given directory.
	Generate(GenerateArgs) error

	// RegenerateCgo: Regenerate cgo
	//
	// Regenerates cgo definitions.
	RegenerateCgo(URIArg) error

	// Tidy: Run go mod tidy
	//
	// Runs `go mod tidy` for a module.
	Tidy(URIArg) error

	// Vendor: Run go mod vendor
	//
	// Runs `go mod vendor` for a module.
	Vendor(URIArg) error

	// UpdateGoSum: Update go.sum
	//
	// Updates the go.sum file for a module.
	UpdateGoSum(URIArg) error

	// CheckUpgrades: Check for upgrades
	//
	// Checks for module upgrades.
	CheckUpgrades(CheckUpgradesArgs) error

	// AddDependency: Add dependency
	//
	// Adds a dependency to the go.mod file for a module.
	AddDependency(DependencyArgs) error

	// UpgradeDependency: Upgrade dependency
	//
	// Upgrades a dependency in the go.mod file for a module.
	UpgradeDependency(DependencyArgs) error

	// RemoveDependency: Remove dependency
	//
	// Removes a dependency from the go.mod file of a module.
	RemoveDependency(RemoveDependencyArgs) error

	// GoGetPackage: go get package
	//
	// Runs `go get` to fetch a package.
	GoGetPackage(GoGetPackageArgs) error

	// ToggleDetails: Toggle gc_details
	//
	// Toggle the calculation of gc annotations.
	ToggleDetails(URIArg) error

	// GenerateGoplsMod: Generate gopls.mod
	//
	// (Re)generate the gopls.mod file for a workspace.
	GenerateGoplsMod(URIArg) error
}

type RunTestsArgs struct {
	// URI is the test file containing the tests to run.
	URI protocol.DocumentURI

	// Tests holds specific test names to run, e.g. TestFoo.
	Tests []string

	// Benchmarks holds specific benchmarks to run, e.g. BenchmarkFoo.
	Benchmarks []string
}

type GenerateArgs struct {
	// URI is any file within the directory to generate. Usually this is the file
	// containing the '//go:generate' directive.
	URI protocol.DocumentURI

	// Recursive controls whether to generate recursively (go generate ./...)
	Recursive bool
}

// TODO(rFindley): document the rest of these once the docgen is fleshed out.

type URIArg struct {
	URI protocol.DocumentURI
}

type CheckUpgradesArgs struct {
	URI     protocol.DocumentURI
	Modules []string
}

type DependencyArgs struct {
	URI        protocol.DocumentURI
	GoCmdArgs  []string
	AddRequire bool
}

type RemoveDependencyArgs struct {
	URI            protocol.DocumentURI
	ModulePath     string
	OnlyDiagnostic bool
}

type GoGetPackageArgs struct {
	URI protocol.DocumentURI
	Pkg string
}
