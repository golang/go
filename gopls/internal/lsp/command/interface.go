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

	"golang.org/x/tools/gopls/internal/govulncheck"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
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

	// EditGoDirective: Run go mod edit -go=version
	//
	// Runs `go mod edit -go=version` for a module.
	EditGoDirective(context.Context, EditGoDirectiveArgs) error

	// UpdateGoSum: Update go.sum
	//
	// Updates the go.sum file for a module.
	UpdateGoSum(context.Context, URIArgs) error

	// CheckUpgrades: Check for upgrades
	//
	// Checks for module upgrades.
	CheckUpgrades(context.Context, CheckUpgradesArgs) error

	// AddDependency: Add a dependency
	//
	// Adds a dependency to the go.mod file for a module.
	AddDependency(context.Context, DependencyArgs) error

	// UpgradeDependency: Upgrade a dependency
	//
	// Upgrades a dependency in the go.mod file for a module.
	UpgradeDependency(context.Context, DependencyArgs) error

	// RemoveDependency: Remove a dependency
	//
	// Removes a dependency from the go.mod file of a module.
	RemoveDependency(context.Context, RemoveDependencyArgs) error

	// ResetGoModDiagnostics: Reset go.mod diagnostics
	//
	// Reset diagnostics in the go.mod file of a module.
	ResetGoModDiagnostics(context.Context, ResetGoModDiagnosticsArgs) error

	// GoGetPackage: go get a package
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

	// ListKnownPackages: List known packages
	//
	// Retrieve a list of packages that are importable from the given URI.
	ListKnownPackages(context.Context, URIArg) (ListKnownPackagesResult, error)

	// ListImports: List imports of a file and its package
	//
	// Retrieve a list of imports in the given Go file, and the package it
	// belongs to.
	ListImports(context.Context, URIArg) (ListImportsResult, error)

	// AddImport: Add an import
	//
	// Ask the server to add an import path to a given Go file.  The method will
	// call applyEdit on the client so that clients don't have to apply the edit
	// themselves.
	AddImport(context.Context, AddImportArgs) error

	// StartDebugging: Start the gopls debug server
	//
	// Start the gopls debug server if it isn't running, and return the debug
	// address.
	StartDebugging(context.Context, DebuggingArgs) (DebuggingResult, error)

	// RunGovulncheck: Run govulncheck.
	//
	// Run vulnerability check (`govulncheck`).
	RunGovulncheck(context.Context, VulncheckArgs) (RunVulncheckResult, error)

	// FetchVulncheckResult: Get known vulncheck result
	//
	// Fetch the result of latest vulnerability check (`govulncheck`).
	FetchVulncheckResult(context.Context, URIArg) (map[protocol.DocumentURI]*govulncheck.Result, error)
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

type EditGoDirectiveArgs struct {
	// Any document URI within the relevant module.
	URI protocol.DocumentURI
	// The version to pass to `go mod edit -go`.
	Version string
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

type ListImportsResult struct {
	// Imports is a list of imports in the requested file.
	Imports []FileImport

	// PackageImports is a list of all imports in the requested file's package.
	PackageImports []PackageImport
}

type FileImport struct {
	// Path is the import path of the import.
	Path string
	// Name is the name of the import, e.g. `foo` in `import foo "strings"`.
	Name string
}

type PackageImport struct {
	// Path is the import path of the import.
	Path string
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

type ResetGoModDiagnosticsArgs struct {
	URIArg

	// Optional: source of the diagnostics to reset.
	// If not set, all resettable go.mod diagnostics will be cleared.
	DiagnosticSource string
}

type VulncheckArgs struct {
	// Any document in the directory from which govulncheck will run.
	URI protocol.DocumentURI

	// Package pattern. E.g. "", ".", "./...".
	Pattern string

	// TODO: -tests
}

// RunVulncheckResult holds the result of asynchronously starting the vulncheck
// command.
type RunVulncheckResult struct {
	// Token holds the progress token for LSP workDone reporting of the vulncheck
	// invocation.
	Token protocol.ProgressToken
}

type VulncheckResult struct {
	Vuln []Vuln

	// TODO: Text string format output?
}

// CallStack models a trace of function calls starting
// with a client function or method and ending with a
// call to a vulnerable symbol.
type CallStack []StackEntry

// StackEntry models an element of a call stack.
type StackEntry struct {
	// See golang.org/x/exp/vulncheck.StackEntry.

	// User-friendly representation of function/method names.
	// e.g. package.funcName, package.(recvType).methodName, ...
	Name string
	URI  protocol.DocumentURI
	Pos  protocol.Position // Start position. (0-based. Column is always 0)
}

// Vuln models an osv.Entry and representative call stacks.
// TODO: deprecate
type Vuln struct {
	// ID is the vulnerability ID (osv.Entry.ID).
	// https://ossf.github.io/osv-schema/#id-modified-fields
	ID string
	// Details is the description of the vulnerability (osv.Entry.Details).
	// https://ossf.github.io/osv-schema/#summary-details-fields
	Details string `json:",omitempty"`
	// Aliases are alternative IDs of the vulnerability.
	// https://ossf.github.io/osv-schema/#aliases-field
	Aliases []string `json:",omitempty"`

	// Symbol is the name of the detected vulnerable function or method.
	// Can be empty if the vulnerability exists in required modules, but no vulnerable symbols are used.
	Symbol string `json:",omitempty"`
	// PkgPath is the package path of the detected Symbol.
	// Can be empty if the vulnerability exists in required modules, but no vulnerable packages are used.
	PkgPath string `json:",omitempty"`
	// ModPath is the module path corresponding to PkgPath.
	// TODO: how do we specify standard library's vulnerability?
	ModPath string `json:",omitempty"`

	// URL is the URL for more info about the information.
	// Either the database specific URL or the one of the URLs
	// included in osv.Entry.References.
	URL string `json:",omitempty"`

	// Current is the current module version.
	CurrentVersion string `json:",omitempty"`

	// Fixed is the minimum module version that contains the fix.
	FixedVersion string `json:",omitempty"`

	// Example call stacks.
	CallStacks []CallStack `json:",omitempty"`

	// Short description of each call stack in CallStacks.
	CallStackSummaries []string `json:",omitempty"`

	// TODO: import graph & module graph.
}
