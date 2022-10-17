// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"context"
	"crypto/sha256"
	"errors"
	"fmt"
	"go/ast"
	"go/scanner"
	"go/token"
	"go/types"
	"io"
	"strings"

	"golang.org/x/mod/modfile"
	"golang.org/x/mod/module"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/gopls/internal/govulncheck"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/imports"
)

// Snapshot represents the current state for the given view.
type Snapshot interface {
	ID() uint64

	// View returns the View associated with this snapshot.
	View() View

	// BackgroundContext returns a context used for all background processing
	// on behalf of this snapshot.
	BackgroundContext() context.Context

	// Fileset returns the Fileset used to parse all the Go files in this snapshot.
	FileSet() *token.FileSet

	// ValidBuildConfiguration returns true if there is some error in the
	// user's workspace. In particular, if they are both outside of a module
	// and their GOPATH.
	ValidBuildConfiguration() bool

	// WriteEnv writes the view-specific environment to the io.Writer.
	WriteEnv(ctx context.Context, w io.Writer) error

	// FindFile returns the FileHandle for the given URI, if it is already
	// in the given snapshot.
	FindFile(uri span.URI) VersionedFileHandle

	// GetVersionedFile returns the VersionedFileHandle for a given URI,
	// initializing it if it is not already part of the snapshot.
	GetVersionedFile(ctx context.Context, uri span.URI) (VersionedFileHandle, error)

	// GetFile returns the FileHandle for a given URI, initializing it if it is
	// not already part of the snapshot.
	GetFile(ctx context.Context, uri span.URI) (FileHandle, error)

	// AwaitInitialized waits until the snapshot's view is initialized.
	AwaitInitialized(ctx context.Context)

	// IsOpen returns whether the editor currently has a file open.
	IsOpen(uri span.URI) bool

	// IgnoredFile reports if a file would be ignored by a `go list` of the whole
	// workspace.
	IgnoredFile(uri span.URI) bool

	// Templates returns the .tmpl files
	Templates() map[span.URI]VersionedFileHandle

	// ParseGo returns the parsed AST for the file.
	// If the file is not available, returns nil and an error.
	ParseGo(ctx context.Context, fh FileHandle, mode ParseMode) (*ParsedGoFile, error)

	// DiagnosePackage returns basic diagnostics, including list, parse, and type errors
	// for pkg, grouped by file.
	DiagnosePackage(ctx context.Context, pkg Package) (map[span.URI][]*Diagnostic, error)

	// Analyze runs the analyses for the given package at this snapshot.
	Analyze(ctx context.Context, pkgID string, analyzers []*Analyzer) ([]*Diagnostic, error)

	// RunGoCommandPiped runs the given `go` command, writing its output
	// to stdout and stderr. Verb, Args, and WorkingDir must be specified.
	//
	// RunGoCommandPiped runs the command serially using gocommand.RunPiped,
	// enforcing that this command executes exclusively to other commands on the
	// server.
	RunGoCommandPiped(ctx context.Context, mode InvocationFlags, inv *gocommand.Invocation, stdout, stderr io.Writer) error

	// RunGoCommandDirect runs the given `go` command. Verb, Args, and
	// WorkingDir must be specified.
	RunGoCommandDirect(ctx context.Context, mode InvocationFlags, inv *gocommand.Invocation) (*bytes.Buffer, error)

	// RunGoCommands runs a series of `go` commands that updates the go.mod
	// and go.sum file for wd, and returns their updated contents.
	RunGoCommands(ctx context.Context, allowNetwork bool, wd string, run func(invoke func(...string) (*bytes.Buffer, error)) error) (bool, []byte, []byte, error)

	// RunProcessEnvFunc runs fn with the process env for this snapshot's view.
	// Note: the process env contains cached module and filesystem state.
	RunProcessEnvFunc(ctx context.Context, fn func(*imports.Options) error) error

	// ModFiles are the go.mod files enclosed in the snapshot's view and known
	// to the snapshot.
	ModFiles() []span.URI

	// ParseMod is used to parse go.mod files.
	ParseMod(ctx context.Context, fh FileHandle) (*ParsedModule, error)

	// ModWhy returns the results of `go mod why` for the module specified by
	// the given go.mod file.
	ModWhy(ctx context.Context, fh FileHandle) (map[string]string, error)

	// ModTidy returns the results of `go mod tidy` for the module specified by
	// the given go.mod file.
	ModTidy(ctx context.Context, pm *ParsedModule) (*TidiedModule, error)

	// GoModForFile returns the URI of the go.mod file for the given URI.
	GoModForFile(uri span.URI) span.URI

	// WorkFile, if non-empty, is the go.work file for the workspace.
	WorkFile() span.URI

	// ParseWork is used to parse go.work files.
	ParseWork(ctx context.Context, fh FileHandle) (*ParsedWorkFile, error)

	// BuiltinFile returns information about the special builtin package.
	BuiltinFile(ctx context.Context) (*ParsedGoFile, error)

	// IsBuiltin reports whether uri is part of the builtin package.
	IsBuiltin(ctx context.Context, uri span.URI) bool

	// PackagesForFile returns an unordered list of packages that contain
	// the file denoted by uri, type checked in the specified mode.
	//
	// If withIntermediateTestVariants is set, the resulting package set includes
	// intermediate test variants.
	PackagesForFile(ctx context.Context, uri span.URI, mode TypecheckMode, withIntermediateTestVariants bool) ([]Package, error)

	// PackageForFile returns a single package that this file belongs to,
	// checked in mode and filtered by the package policy.
	PackageForFile(ctx context.Context, uri span.URI, mode TypecheckMode, selectPackage PackageFilter) (Package, error)

	// GetActiveReverseDeps returns the active files belonging to the reverse
	// dependencies of this file's package, checked in TypecheckWorkspace mode.
	GetReverseDependencies(ctx context.Context, id string) ([]Package, error)

	// CachedImportPaths returns all the imported packages loaded in this
	// snapshot, indexed by their package path (not import path, despite the name)
	// and checked in TypecheckWorkspace mode.
	CachedImportPaths(ctx context.Context) (map[string]Package, error)

	// KnownPackages returns all the packages loaded in this snapshot, checked
	// in TypecheckWorkspace mode.
	KnownPackages(ctx context.Context) ([]Package, error)

	// ActivePackages returns the packages considered 'active' in the workspace.
	//
	// In normal memory mode, this is all workspace packages. In degraded memory
	// mode, this is just the reverse transitive closure of open packages.
	ActivePackages(ctx context.Context) ([]Package, error)

	// AllValidMetadata returns all valid metadata loaded for the snapshot.
	AllValidMetadata(ctx context.Context) ([]Metadata, error)

	// WorkspacePackageByID returns the workspace package with id, type checked
	// in 'workspace' mode.
	WorkspacePackageByID(ctx context.Context, id string) (Package, error)

	// Symbols returns all symbols in the snapshot.
	Symbols(ctx context.Context) map[span.URI][]Symbol

	// Metadata returns package metadata associated with the given file URI.
	MetadataForFile(ctx context.Context, uri span.URI) ([]Metadata, error)

	// GetCriticalError returns any critical errors in the workspace.
	GetCriticalError(ctx context.Context) *CriticalError

	// BuildGoplsMod generates a go.mod file for all modules in the workspace.
	// It bypasses any existing gopls.mod.
	BuildGoplsMod(ctx context.Context) (*modfile.File, error)
}

// PackageFilter sets how a package is filtered out from a set of packages
// containing a given file.
type PackageFilter int

const (
	// NarrowestPackage picks the "narrowest" package for a given file.
	// By "narrowest" package, we mean the package with the fewest number of
	// files that includes the given file. This solves the problem of test
	// variants, as the test will have more files than the non-test package.
	NarrowestPackage PackageFilter = iota

	// WidestPackage returns the Package containing the most files.
	// This is useful for something like diagnostics, where we'd prefer to
	// offer diagnostics for as many files as possible.
	WidestPackage
)

// InvocationFlags represents the settings of a particular go command invocation.
// It is a mode, plus a set of flag bits.
type InvocationFlags int

const (
	// Normal is appropriate for commands that might be run by a user and don't
	// deliberately modify go.mod files, e.g. `go test`.
	Normal InvocationFlags = iota
	// WriteTemporaryModFile is for commands that need information from a
	// modified version of the user's go.mod file, e.g. `go mod tidy` used to
	// generate diagnostics.
	WriteTemporaryModFile
	// LoadWorkspace is for packages.Load, and other operations that should
	// consider the whole workspace at once.
	LoadWorkspace

	// AllowNetwork is a flag bit that indicates the invocation should be
	// allowed to access the network.
	AllowNetwork InvocationFlags = 1 << 10
)

func (m InvocationFlags) Mode() InvocationFlags {
	return m & (AllowNetwork - 1)
}

func (m InvocationFlags) AllowNetwork() bool {
	return m&AllowNetwork != 0
}

// View represents a single workspace.
// This is the level at which we maintain configuration like working directory
// and build tags.
type View interface {
	// Name returns the name this view was constructed with.
	Name() string

	// Folder returns the folder with which this view was created.
	Folder() span.URI

	// Shutdown closes this view, and detaches it from its session.
	Shutdown(ctx context.Context)

	// Options returns a copy of the Options for this view.
	Options() *Options

	// SetOptions sets the options of this view to new values.
	// Calling this may cause the view to be invalidated and a replacement view
	// added to the session. If so the new view will be returned, otherwise the
	// original one will be.
	SetOptions(context.Context, *Options) (View, error)

	// Snapshot returns the current snapshot for the view, and a
	// release function that must be called when the Snapshot is
	// no longer needed.
	Snapshot(ctx context.Context) (Snapshot, func())

	// IsGoPrivatePath reports whether target is a private import path, as identified
	// by the GOPRIVATE environment variable.
	IsGoPrivatePath(path string) bool

	// ModuleUpgrades returns known module upgrades for the dependencies of
	// modfile.
	ModuleUpgrades(modfile span.URI) map[string]string

	// RegisterModuleUpgrades registers that upgrades exist for the given modules
	// required by modfile.
	RegisterModuleUpgrades(modfile span.URI, upgrades map[string]string)

	// ClearModuleUpgrades clears all upgrades for the modules in modfile.
	ClearModuleUpgrades(modfile span.URI)

	// Vulnerabilites returns known vulnerabilities for the given modfile.
	// TODO(suzmue): replace command.Vuln with a different type, maybe
	// https://pkg.go.dev/golang.org/x/vuln/cmd/govulncheck/govulnchecklib#Summary?
	Vulnerabilities(modfile span.URI) []govulncheck.Vuln

	// SetVulnerabilities resets the list of vulnerabilites that exists for the given modules
	// required by modfile.
	SetVulnerabilities(modfile span.URI, vulnerabilities []govulncheck.Vuln)

	// FileKind returns the type of a file
	FileKind(FileHandle) FileKind

	// GoVersion returns the configured Go version for this view.
	GoVersion() int
}

// A FileSource maps uris to FileHandles. This abstraction exists both for
// testability, and so that algorithms can be run equally on session and
// snapshot files.
type FileSource interface {
	// GetFile returns the FileHandle for a given URI.
	GetFile(ctx context.Context, uri span.URI) (FileHandle, error)
}

// A ParsedGoFile contains the results of parsing a Go file.
type ParsedGoFile struct {
	URI  span.URI
	Mode ParseMode
	File *ast.File
	Tok  *token.File
	// Source code used to build the AST. It may be different from the
	// actual content of the file if we have fixed the AST.
	Src      []byte
	Fixed    bool
	Mapper   *protocol.ColumnMapper
	ParseErr scanner.ErrorList
}

// A ParsedModule contains the results of parsing a go.mod file.
type ParsedModule struct {
	URI         span.URI
	File        *modfile.File
	Mapper      *protocol.ColumnMapper
	ParseErrors []*Diagnostic
}

// A ParsedWorkFile contains the results of parsing a go.work file.
type ParsedWorkFile struct {
	URI         span.URI
	File        *modfile.WorkFile
	Mapper      *protocol.ColumnMapper
	ParseErrors []*Diagnostic
}

// A TidiedModule contains the results of running `go mod tidy` on a module.
type TidiedModule struct {
	// Diagnostics representing changes made by `go mod tidy`.
	Diagnostics []*Diagnostic
	// The bytes of the go.mod file after it was tidied.
	TidiedContent []byte
}

// Metadata represents package metadata retrieved from go/packages.
//
// TODO(rfindley): move the strongly typed strings from the cache package here.
type Metadata interface {
	// PackageID is the unique package id.
	PackageID() string

	// PackageName is the package name.
	PackageName() string

	// PackagePath is the package path.
	PackagePath() string

	// ModuleInfo returns the go/packages module information for the given package.
	ModuleInfo() *packages.Module
}

var ErrViewExists = errors.New("view already exists for session")

// Overlay is the type for a file held in memory on a session.
type Overlay interface {
	Kind() FileKind
	VersionedFileHandle
}

// FileModification represents a modification to a file.
type FileModification struct {
	URI    span.URI
	Action FileAction

	// OnDisk is true if a watched file is changed on disk.
	// If true, Version will be -1 and Text will be nil.
	OnDisk bool

	// Version will be -1 and Text will be nil when they are not supplied,
	// specifically on textDocument/didClose and for on-disk changes.
	Version int32
	Text    []byte

	// LanguageID is only sent from the language client on textDocument/didOpen.
	LanguageID string
}

type FileAction int

const (
	UnknownFileAction = FileAction(iota)
	Open
	Change
	Close
	Save
	Create
	Delete
	InvalidateMetadata
)

func (a FileAction) String() string {
	switch a {
	case Open:
		return "Open"
	case Change:
		return "Change"
	case Close:
		return "Close"
	case Save:
		return "Save"
	case Create:
		return "Create"
	case Delete:
		return "Delete"
	case InvalidateMetadata:
		return "InvalidateMetadata"
	default:
		return "Unknown"
	}
}

var ErrTmpModfileUnsupported = errors.New("-modfile is unsupported for this Go version")
var ErrNoModOnDisk = errors.New("go.mod file is not on disk")

func IsNonFatalGoModError(err error) bool {
	return err == ErrTmpModfileUnsupported || err == ErrNoModOnDisk
}

// ParseMode controls the content of the AST produced when parsing a source file.
type ParseMode int

const (
	// ParseHeader specifies that the main package declaration and imports are needed.
	// This is the mode used when attempting to examine the package graph structure.
	ParseHeader ParseMode = iota

	// ParseExported specifies that the package is used only as a dependency,
	// and only its exported declarations are needed. More may be included if
	// necessary to avoid type errors.
	ParseExported

	// ParseFull specifies the full AST is needed.
	// This is used for files of direct interest where the entire contents must
	// be considered.
	ParseFull
)

// AllParseModes contains all possible values of ParseMode.
// It is used for cache invalidation on a file content change.
var AllParseModes = []ParseMode{ParseHeader, ParseExported, ParseFull}

// TypecheckMode controls what kind of parsing should be done (see ParseMode)
// while type checking a package.
type TypecheckMode int

const (
	// TypecheckFull means to use ParseFull.
	TypecheckFull TypecheckMode = iota
	// TypecheckWorkspace means to use ParseFull for workspace packages, and
	// ParseExported for others.
	TypecheckWorkspace
)

type VersionedFileHandle interface {
	FileHandle
	Version() int32
	Session() string

	// LSPIdentity returns the version identity of a file.
	VersionedFileIdentity() VersionedFileIdentity
}

type VersionedFileIdentity struct {
	URI span.URI

	// SessionID is the ID of the LSP session.
	SessionID string

	// Version is the version of the file, as specified by the client. It should
	// only be set in combination with SessionID.
	Version int32
}

// FileHandle represents a handle to a specific version of a single file.
type FileHandle interface {
	URI() span.URI

	// FileIdentity returns a FileIdentity for the file, even if there was an
	// error reading it.
	FileIdentity() FileIdentity
	// Read reads the contents of a file.
	// If the file is not available, returns a nil slice and an error.
	Read() ([]byte, error)
	// Saved reports whether the file has the same content on disk.
	Saved() bool
}

// A Hash is a cryptographic digest of the contents of a file.
// (Although at 32B it is larger than a 16B string header, it is smaller
// and has better locality than the string header + 64B of hex digits.)
type Hash [sha256.Size]byte

// HashOf returns the hash of some data.
func HashOf(data []byte) Hash {
	return Hash(sha256.Sum256(data))
}

// Hashf returns the hash of a printf-formatted string.
func Hashf(format string, args ...interface{}) Hash {
	// Although this looks alloc-heavy, it is faster than using
	// Fprintf on sha256.New() because the allocations don't escape.
	return HashOf([]byte(fmt.Sprintf(format, args...)))
}

// String returns the digest as a string of hex digits.
func (h Hash) String() string {
	return fmt.Sprintf("%64x", [sha256.Size]byte(h))
}

// Less returns true if the given hash is less than the other.
func (h Hash) Less(other Hash) bool {
	return bytes.Compare(h[:], other[:]) < 0
}

// XORWith updates *h to *h XOR h2.
func (h *Hash) XORWith(h2 Hash) {
	// Small enough that we don't need crypto/subtle.XORBytes.
	for i := range h {
		h[i] ^= h2[i]
	}
}

// FileIdentity uniquely identifies a file at a version from a FileSystem.
type FileIdentity struct {
	URI  span.URI
	Hash Hash // digest of file contents
}

func (id FileIdentity) String() string {
	return fmt.Sprintf("%s%s", id.URI, id.Hash)
}

// FileKind describes the kind of the file in question.
// It can be one of Go,mod, Sum, or Tmpl.
type FileKind int

const (
	// UnknownKind is a file type we don't know about.
	UnknownKind = FileKind(iota)

	// Go is a normal go source file.
	Go
	// Mod is a go.mod file.
	Mod
	// Sum is a go.sum file.
	Sum
	// Tmpl is a template file.
	Tmpl
	// Work is a go.work file.
	Work
)

// Analyzer represents a go/analysis analyzer with some boolean properties
// that let the user know how to use the analyzer.
type Analyzer struct {
	Analyzer *analysis.Analyzer

	// Enabled reports whether the analyzer is enabled. This value can be
	// configured per-analysis in user settings. For staticcheck analyzers,
	// the value of the Staticcheck setting overrides this field.
	Enabled bool

	// Fix is the name of the suggested fix name used to invoke the suggested
	// fixes for the analyzer. It is non-empty if we expect this analyzer to
	// provide its fix separately from its diagnostics. That is, we should apply
	// the analyzer's suggested fixes through a Command, not a TextEdit.
	Fix string

	// ActionKind is the kind of code action this analyzer produces. If
	// unspecified the type defaults to quickfix.
	ActionKind []protocol.CodeActionKind

	// Severity is the severity set for diagnostics reported by this
	// analyzer. If left unset it defaults to Warning.
	Severity protocol.DiagnosticSeverity
}

func (a Analyzer) IsEnabled(view View) bool {
	// Staticcheck analyzers can only be enabled when staticcheck is on.
	if _, ok := view.Options().StaticcheckAnalyzers[a.Analyzer.Name]; ok {
		if !view.Options().Staticcheck {
			return false
		}
	}
	if enabled, ok := view.Options().Analyses[a.Analyzer.Name]; ok {
		return enabled
	}
	return a.Enabled
}

// Package represents a Go package that has been type-checked. It maintains
// only the relevant fields of a *go/packages.Package.
type Package interface {
	ID() string      // logically a cache.PackageID
	Name() string    // logically a cache.PackageName
	PkgPath() string // logically a cache.PackagePath
	CompiledGoFiles() []*ParsedGoFile
	File(uri span.URI) (*ParsedGoFile, error)
	GetSyntax() []*ast.File
	GetTypes() *types.Package
	GetTypesInfo() *types.Info
	GetTypesSizes() types.Sizes
	ForTest() string
	DirectDep(packagePath string) (Package, error)        // logically a cache.PackagePath
	ResolveImportPath(importPath string) (Package, error) // logically a cache.ImportPath
	MissingDependencies() []string                        // unordered; logically cache.ImportPaths
	Imports() []Package
	Version() *module.Version
	HasListOrParseErrors() bool
	HasTypeErrors() bool
	ParseMode() ParseMode
}

// A CriticalError is a workspace-wide error that generally prevents gopls from
// functioning correctly. In the presence of critical errors, other diagnostics
// in the workspace may not make sense.
type CriticalError struct {
	// MainError is the primary error. Must be non-nil.
	MainError error

	// Diagnostics contains any supplemental (structured) diagnostics.
	Diagnostics []*Diagnostic
}

// An Diagnostic corresponds to an LSP Diagnostic.
// https://microsoft.github.io/language-server-protocol/specification#diagnostic
type Diagnostic struct {
	URI      span.URI
	Range    protocol.Range
	Severity protocol.DiagnosticSeverity
	Code     string
	CodeHref string

	// Source is a human-readable description of the source of the error.
	// Diagnostics generated by an analysis.Analyzer set it to Analyzer.Name.
	Source DiagnosticSource

	Message string

	Tags    []protocol.DiagnosticTag
	Related []RelatedInformation

	// Fields below are used internally to generate quick fixes. They aren't
	// part of the LSP spec and don't leave the server.
	SuggestedFixes []SuggestedFix
	Analyzer       *Analyzer
}

func (d *Diagnostic) String() string {
	return fmt.Sprintf("%v: %s", d.Range, d.Message)
}

type DiagnosticSource string

const (
	UnknownError             DiagnosticSource = "<Unknown source>"
	ListError                DiagnosticSource = "go list"
	ParseError               DiagnosticSource = "syntax"
	TypeError                DiagnosticSource = "compiler"
	ModTidyError             DiagnosticSource = "go mod tidy"
	OptimizationDetailsError DiagnosticSource = "optimizer details"
	UpgradeNotification      DiagnosticSource = "upgrade available"
	Vulncheck                DiagnosticSource = "govulncheck"
	TemplateError            DiagnosticSource = "template"
	WorkFileError            DiagnosticSource = "go.work file"
)

func AnalyzerErrorKind(name string) DiagnosticSource {
	return DiagnosticSource(name)
}

// WorkspaceModuleVersion is the nonexistent pseudoversion suffix used in the
// construction of the workspace module. It is exported so that we can make
// sure not to show this version to end users in error messages, to avoid
// confusion.
// The major version is not included, as that depends on the module path.
//
// If workspace module A is dependent on workspace module B, we need our
// nonexistent version to be greater than the version A mentions.
// Otherwise, the go command will try to update to that version. Use a very
// high minor version to make that more likely.
const workspaceModuleVersion = ".9999999.0-goplsworkspace"

func IsWorkspaceModuleVersion(version string) bool {
	return strings.HasSuffix(version, workspaceModuleVersion)
}

func WorkspaceModuleVersion(majorVersion string) string {
	// Use the highest compatible major version to avoid unwanted upgrades.
	// See the comment on workspaceModuleVersion.
	if majorVersion == "v0" {
		majorVersion = "v1"
	}
	return majorVersion + workspaceModuleVersion
}
