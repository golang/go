// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/packagesinternal"
	"golang.org/x/tools/internal/span"
)

// Snapshot represents the current state for the given view.
type Snapshot interface {
	ID() uint64

	// View returns the View associated with this snapshot.
	View() View

	// Config returns the configuration for the view.
	Config(ctx context.Context) *packages.Config

	// GetFile returns the file object for a given URI, initializing it
	// if it is not already part of the view.
	GetFile(uri span.URI) (FileHandle, error)

	// IsOpen returns whether the editor currently has a file open.
	IsOpen(uri span.URI) bool

	// IsSaved returns whether the contents are saved on disk or not.
	IsSaved(uri span.URI) bool

	// Analyze runs the analyses for the given package at this snapshot.
	Analyze(ctx context.Context, id string, analyzers []*analysis.Analyzer) ([]*Error, error)

	// FindAnalysisError returns the analysis error represented by the diagnostic.
	// This is used to get the SuggestedFixes associated with that error.
	FindAnalysisError(ctx context.Context, pkgID, analyzerName, msg string, rng protocol.Range) (*Error, error)

	// ModTidyHandle returns a ModTidyHandle for the given go.mod file handle.
	// This function can have no data or error if there is no modfile detected.
	ModTidyHandle(ctx context.Context, fh FileHandle) (ModTidyHandle, error)

	// ModHandle returns a ModHandle for the passed in go.mod file handle.
	// This function can have no data if there is no modfile detected.
	ModHandle(ctx context.Context, fh FileHandle) ModHandle

	// PackageHandles returns the PackageHandles for the packages that this file
	// belongs to.
	PackageHandles(ctx context.Context, fh FileHandle) ([]PackageHandle, error)

	// GetActiveReverseDeps returns the active files belonging to the reverse
	// dependencies of this file's package.
	GetReverseDependencies(ctx context.Context, id string) ([]PackageHandle, error)

	// CachedImportPaths returns all the imported packages loaded in this snapshot,
	// indexed by their import path.
	CachedImportPaths(ctx context.Context) (map[string]Package, error)

	// KnownPackages returns all the packages loaded in this snapshot.
	// Workspace packages may be parsed in ParseFull mode, whereas transitive
	// dependencies will be in ParseExported mode.
	KnownPackages(ctx context.Context) ([]PackageHandle, error)

	// WorkspacePackages returns the PackageHandles for the snapshot's
	// top-level packages.
	WorkspacePackages(ctx context.Context) ([]PackageHandle, error)
}

// PackageHandle represents a handle to a specific version of a package.
// It is uniquely defined by the file handles that make up the package.
type PackageHandle interface {
	// ID returns the ID of the package associated with the PackageHandle.
	ID() string

	// CompiledGoFiles returns the ParseGoHandles composing the package.
	CompiledGoFiles() []ParseGoHandle

	// Check returns the type-checked Package for the PackageHandle.
	Check(ctx context.Context) (Package, error)

	// Cached returns the Package for the PackageHandle if it has already been stored.
	Cached() (Package, error)

	// MissingDependencies reports any unresolved imports.
	MissingDependencies() []string
}

// View represents a single workspace.
// This is the level at which we maintain configuration like working directory
// and build tags.
type View interface {
	// Session returns the session that created this view.
	Session() Session

	// Name returns the name this view was constructed with.
	Name() string

	// Folder returns the root folder for this view.
	Folder() span.URI

	// ModFiles returns the URIs of the go.mod files attached to the view associated with this snapshot.
	ModFiles() (span.URI, span.URI)

	// LookupBuiltin returns the go/ast.Object for the given name in the builtin package.
	LookupBuiltin(ctx context.Context, name string) (*ast.Object, error)

	// BackgroundContext returns a context used for all background processing
	// on behalf of this view.
	BackgroundContext() context.Context

	// Shutdown closes this view, and detaches it from it's session.
	Shutdown(ctx context.Context)

	// Ignore returns true if this file should be ignored by this view.
	Ignore(span.URI) bool

	// RunProcessEnvFunc runs fn with the process env for this snapshot's view.
	// Note: the process env contains cached module and filesystem state.
	RunProcessEnvFunc(ctx context.Context, fn func(*imports.Options) error) error

	// Options returns a copy of the Options for this view.
	Options() Options

	// SetOptions sets the options of this view to new values.
	// Calling this may cause the view to be invalidated and a replacement view
	// added to the session. If so the new view will be returned, otherwise the
	// original one will be.
	SetOptions(context.Context, Options) (View, error)

	// Snapshot returns the current snapshot for the view.
	Snapshot() Snapshot

	// Rebuild rebuilds the current view, replacing the original view in its session.
	Rebuild(ctx context.Context) (Snapshot, error)

	// InvalidBuildConfiguration returns true if there is some error in the
	// user's workspace. In particular, if they are both outside of a module
	// and their GOPATH.
	ValidBuildConfiguration() bool
}

// Session represents a single connection from a client.
// This is the level at which things like open files are maintained on behalf
// of the client.
// A session may have many active views at any given time.
type Session interface {
	// NewView creates a new View and returns it.
	NewView(ctx context.Context, name string, folder span.URI, options Options) (View, Snapshot, error)

	// Cache returns the cache that created this session.
	Cache() Cache

	// View returns a view with a matching name, if the session has one.
	View(name string) View

	// ViewOf returns a view corresponding to the given URI.
	ViewOf(uri span.URI) (View, error)

	// Views returns the set of active views built by this session.
	Views() []View

	// Shutdown the session and all views it has created.
	Shutdown(ctx context.Context)

	// A FileSystem prefers the contents from overlays, and falls back to the
	// content from the underlying cache if no overlay is present.
	FileSystem

	// DidModifyFile reports a file modification to the session.
	// It returns the resulting snapshots, a guaranteed one per view.
	DidModifyFiles(ctx context.Context, changes []FileModification) ([]Snapshot, error)

	// Options returns a copy of the SessionOptions for this session.
	Options() Options

	// SetOptions sets the options of this session to new values.
	SetOptions(Options)
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
	Version float64
	Text    []byte

	// LanguageID is only sent from the language client on textDocument/didOpen.
	LanguageID string
}

type FileAction int

const (
	Open = FileAction(iota)
	Change
	Close
	Save
	Create
	Delete
	UnknownFileAction
)

// Cache abstracts the core logic of dealing with the environment from the
// higher level logic that processes the information to produce results.
// The cache provides access to files and their contents, so the source
// package does not directly access the file system.
// A single cache is intended to be process wide, and is the primary point of
// sharing between all consumers.
// A cache may have many active sessions at any given time.
type Cache interface {
	// A FileSystem that reads file contents from external storage.
	FileSystem

	// FileSet returns the shared fileset used by all files in the system.
	FileSet() *token.FileSet

	// ParseGoHandle returns a ParseGoHandle for the given file handle.
	ParseGoHandle(fh FileHandle, mode ParseMode) ParseGoHandle
}

// FileSystem is the interface to something that provides file contents.
type FileSystem interface {
	// GetFile returns a handle for the specified file.
	GetFile(uri span.URI) FileHandle
}

// ParseGoHandle represents a handle to the AST for a file.
type ParseGoHandle interface {
	// File returns a file handle for which to get the AST.
	File() FileHandle

	// Mode returns the parse mode of this handle.
	Mode() ParseMode

	// Parse returns the parsed AST for the file.
	// If the file is not available, returns nil and an error.
	Parse(ctx context.Context) (file *ast.File, src []byte, m *protocol.ColumnMapper, parseErr error, err error)

	// Cached returns the AST for this handle, if it has already been stored.
	Cached() (file *ast.File, src []byte, m *protocol.ColumnMapper, parseErr error, err error)
}

// ModHandle represents a handle to the modfile for a go.mod.
type ModHandle interface {
	// File returns a file handle for which to get the modfile.
	File() FileHandle

	// Parse returns the parsed modfile and a mapper for the go.mod file.
	// If the file is not available, returns nil and an error.
	Parse(ctx context.Context) (*modfile.File, *protocol.ColumnMapper, error)

	// Upgrades returns the parsed modfile, a mapper, and any dependency upgrades
	// for the go.mod file. Note that this will only work if the go.mod is the view's go.mod.
	// If the file is not available, returns nil and an error.
	Upgrades(ctx context.Context) (*modfile.File, *protocol.ColumnMapper, map[string]string, error)

	// Why returns the parsed modfile, a mapper, and any explanations why a dependency should be
	// in the go.mod file. Note that this will only work if the go.mod is the view's go.mod.
	// If the file is not available, returns nil and an error.
	Why(ctx context.Context) (*modfile.File, *protocol.ColumnMapper, map[string]string, error)
}

// ModTidyHandle represents a handle to the modfile for the view.
// Specifically for the purpose of getting diagnostics by running "go mod tidy".
type ModTidyHandle interface {
	// File returns a file handle for which to get the modfile.
	File() FileHandle

	// Tidy returns the parsed modfile, a mapper, and "go mod tidy" errors
	// for the go.mod file. If the file is not available, returns nil and an error.
	Tidy(ctx context.Context) (*modfile.File, *protocol.ColumnMapper, map[string]*modfile.Require, []Error, error)
}

// ParseMode controls the content of the AST produced when parsing a source file.
type ParseMode int

const (
	// ParseHeader specifies that the main package declaration and imports are needed.
	// This is the mode used when attempting to examine the package graph structure.
	ParseHeader = ParseMode(iota)

	// ParseExported specifies that the public symbols are needed, but things like
	// private symbols and function bodies are not.
	// This mode is used for things where a package is being consumed only as a
	// dependency.
	ParseExported

	// ParseFull specifies the full AST is needed.
	// This is used for files of direct interest where the entire contents must
	// be considered.
	ParseFull
)

// FileHandle represents a handle to a specific version of a single file from
// a specific file system.
type FileHandle interface {
	// FileSystem returns the file system this handle was acquired from.
	FileSystem() FileSystem

	// Identity returns the FileIdentity for the file.
	Identity() FileIdentity

	// Read reads the contents of a file and returns it along with its hash value.
	// If the file is not available, returns a nil slice and an error.
	Read(ctx context.Context) ([]byte, string, error)
}

// FileIdentity uniquely identifies a file at a version from a FileSystem.
type FileIdentity struct {
	URI span.URI

	// Version is the version of the file, as specified by the client.
	Version float64

	// Identifier represents a unique identifier for the file.
	// It could be a file's modification time or its SHA1 hash if it is not on disk.
	Identifier string

	// Kind is the file's kind.
	Kind FileKind
}

func (fileID FileIdentity) String() string {
	// Version is not part of the FileIdentity string,
	// as it can remain change even if the file does not.
	return fmt.Sprintf("%s%s%s", fileID.URI, fileID.Identifier, fileID.Kind)
}

// FileKind describes the kind of the file in question.
// It can be one of Go, mod, or sum.
type FileKind int

const (
	Go = FileKind(iota)
	Mod
	Sum
	UnknownKind
)

// Package represents a Go package that has been type-checked. It maintains
// only the relevant fields of a *go/packages.Package.
type Package interface {
	ID() string
	PkgPath() string
	CompiledGoFiles() []ParseGoHandle
	File(uri span.URI) (ParseGoHandle, error)
	GetSyntax() []*ast.File
	GetErrors() []*Error
	GetTypes() *types.Package
	GetTypesInfo() *types.Info
	GetTypesSizes() types.Sizes
	IsIllTyped() bool
	ForTest() string
	GetImport(pkgPath string) (Package, error)
	Imports() []Package
	Module() *packagesinternal.Module
}

type Error struct {
	URI            span.URI
	Range          protocol.Range
	Kind           ErrorKind
	Message        string
	Category       string // only used by analysis errors so far
	SuggestedFixes []SuggestedFix
	Related        []RelatedInformation
}

type ErrorKind int

const (
	UnknownError = ErrorKind(iota)
	ListError
	ParseError
	TypeError
	Analysis
)

func (e *Error) Error() string {
	return fmt.Sprintf("%s:%s: %s", e.URI, e.Range, e.Message)
}
