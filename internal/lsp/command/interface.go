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

//go:generate go run -tags=generate generate.go

import (
	"context"

	"golang.org/x/tools/internal/lsp/protocol"
)

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
	// ApplyFix: Apply a fix
	//
	// Applies a fix to a region of source code.
	ApplyFix(context.Context, ApplyFixArgs) error
	// Test: Run test(s) (legacy)
	//
	// Runs `go test` for a specific set of test or benchmark functions.
	Test(context.Context, protocol.DocumentURI, []string, []string) error

	// TODO: deprecate Test in favor of RunTests below.

	// Test: Run test(s)
	//
	// Runs `go test` for a specific set of test or benchmark functions.
	RunTests(context.Context, RunTestsArgs) error

	// Generate: Run go generate
	//
	// Runs `go generate` for a given directory.
	Generate(context.Context, GenerateArgs) error

	// RegenerateCgo: Regenerate cgo
	//
	// Regenerates cgo definitions.
	RegenerateCgo(context.Context, URIArg) error

	// Tidy: Run go mod tidy
	//
	// Runs `go mod tidy` for a module.
	Tidy(context.Context, URIArgs) error

	// Vendor: Run go mod vendor
	//
	// Runs `go mod vendor` for a module.
	Vendor(context.Context, URIArg) error

	// UpdateGoSum: Update go.sum
	//
	// Updates the go.sum file for a module.
	UpdateGoSum(context.Context, URIArgs) error

	// CheckUpgrades: Check for upgrades
	//
	// Checks for module upgrades.
	CheckUpgrades(context.Context, CheckUpgradesArgs) error

	// AddDependency: Add dependency
	//
	// Adds a dependency to the go.mod file for a module.
	AddDependency(context.Context, DependencyArgs) error

	// UpgradeDependency: Upgrade dependency
	//
	// Upgrades a dependency in the go.mod file for a module.
	UpgradeDependency(context.Context, DependencyArgs) error

	// RemoveDependency: Remove dependency
	//
	// Removes a dependency from the go.mod file of a module.
	RemoveDependency(context.Context, RemoveDependencyArgs) error

	// GoGetPackage: go get package
	//
	// Runs `go get` to fetch a package.
	GoGetPackage(context.Context, GoGetPackageArgs) error

	// GCDetails: Toggle gc_details
	//
	// Toggle the calculation of gc annotations.
	GCDetails(context.Context, protocol.DocumentURI) error

	// TODO: deprecate GCDetails in favor of ToggleGCDetails below.

	// ToggleGCDetails: Toggle gc_details
	//
	// Toggle the calculation of gc annotations.
	ToggleGCDetails(context.Context, URIArg) error

	// GenerateGoplsMod: Generate gopls.mod
	//
	// (Re)generate the gopls.mod file for a workspace.
	GenerateGoplsMod(context.Context, URIArg) error

	ListKnownPackages(context.Context, URIArg) (ListKnownPackagesResult, error)

	AddImport(context.Context, AddImportArgs) (AddImportResult, error)
}

type RunTestsArgs struct {
	// The test file containing the tests to run.
	URI protocol.DocumentURI

	// Specific test names to run, e.g. TestFoo.
	Tests []string

	// Specific benchmarks to run, e.g. BenchmarkFoo.
	Benchmarks []string
}

type GenerateArgs struct {
	// URI for the directory to generate.
	Dir protocol.DocumentURI

	// Whether to generate recursively (go generate ./...)
	Recursive bool
}

// TODO(rFindley): document the rest of these once the docgen is fleshed out.

type ApplyFixArgs struct {
	// The fix to apply.
	Fix string
	// The file URI for the document to fix.
	URI protocol.DocumentURI
	// The document range to scan for fixes.
	Range protocol.Range
}

type URIArg struct {
	// The file URI.
	URI protocol.DocumentURI
}

type URIArgs struct {
	// The file URIs.
	URIs []protocol.DocumentURI
}

type CheckUpgradesArgs struct {
	// The go.mod file URI.
	URI protocol.DocumentURI
	// The modules to check.
	Modules []string
}

type DependencyArgs struct {
	// The go.mod file URI.
	URI protocol.DocumentURI
	// Additional args to pass to the go command.
	GoCmdArgs []string
	// Whether to add a require directive.
	AddRequire bool
}

type RemoveDependencyArgs struct {
	// The go.mod file URI.
	URI protocol.DocumentURI
	// The module path to remove.
	ModulePath     string
	OnlyDiagnostic bool
}

type GoGetPackageArgs struct {
	// Any document URI within the relevant module.
	URI protocol.DocumentURI
	// The package to go get.
	Pkg        string
	AddRequire bool
}

// TODO (Marwan): document :)

type AddImportArgs struct {
	ImportPath string
	URI        protocol.DocumentURI
}

type AddImportResult struct {
	Edits []protocol.TextDocumentEdit
}

type ListKnownPackagesResult struct {
	Packages []string
}
