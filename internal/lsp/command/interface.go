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

	// ListKnownPackages: retrieves a list of packages
	// that are importable from the given URI.
	ListKnownPackages(context.Context, URIArg) (ListKnownPackagesResult, error)

	// AddImport: asks the server to add an import path to a given Go file.
	// The method will call applyEdit on the client so that clients don't have
	// to apply the edit themselves.
	AddImport(context.Context, AddImportArgs) error

	WorkspaceMetadata(context.Context) (WorkspaceMetadataResult, error)

	StartDebugging(context.Context, DebuggingArgs) (DebuggingResult, error)
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

type AddImportArgs struct {
	// ImportPath is the target import path that should
	// be added to the URI file
	ImportPath string
	// URI is the file that the ImportPath should be
	// added to
	URI protocol.DocumentURI
}

type ListKnownPackagesResult struct {
	// Packages is a list of packages relative
	// to the URIArg passed by the command request.
	// In other words, it omits paths that are already
	// imported or cannot be imported due to compiler
	// restrictions.
	Packages []string
}

type WorkspaceMetadataArgs struct {
}

type WorkspaceMetadataResult struct {
	Workspaces []Workspace
}

type Workspace struct {
	Name      string
	ModuleDir string
}

type DebuggingArgs struct {
	// Optional: the address (including port) for the debug server to listen on.
	// If not provided, the debug server will bind to "localhost:0", and the
	// full debug URL will be contained in the result.
	//
	// If there is more than one gopls instance along the serving path (i.e. you
	// are using a daemon), each gopls instance will attempt to start debugging.
	// If Addr specifies a port, only the daemon will be able to bind to that
	// port, and each intermediate gopls instance will fail to start debugging.
	// For this reason it is recommended not to specify a port (or equivalently,
	// to specify ":0").
	//
	// If the server was already debugging this field has no effect, and the
	// result will contain the previously configured debug URL(s).
	Addr string
}

type DebuggingResult struct {
	// The URLs to use to access the debug servers, for all gopls instances in
	// the serving path. For the common case of a single gopls instance (i.e. no
	// daemon), this will be exactly one address.
	//
	// In the case of one or more gopls instances forwarding the LSP to a daemon,
	// URLs will contain debug addresses for each server in the serving path, in
	// serving order. The daemon debug address will be the last entry in the
	// slice. If any intermediate gopls instance fails to start debugging, no
	// error will be returned but the debug URL for that server in the URLs slice
	// will be empty.
	URLs []string
}
