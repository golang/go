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

	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
)

// Snapshot represents the current state for the given view.
type Snapshot interface {
	// View returns the View associated with this snapshot.
	View() View

	// Handle returns the FileHandle for the given file.
	Handle(ctx context.Context, f File) FileHandle

	// Analyze runs the analyses for the given package at this snapshot.
	Analyze(ctx context.Context, id string, analyzers []*analysis.Analyzer) ([]*Error, error)

	// FindAnalysisError returns the analysis error represented by the diagnostic.
	// This is used to get the SuggestedFixes associated with that error.
	FindAnalysisError(ctx context.Context, id string, diag protocol.Diagnostic) (*Error, error)

	// PackageHandle returns the CheckPackageHandle for the given package ID.
	PackageHandle(ctx context.Context, id string) (CheckPackageHandle, error)

	// PackageHandles returns the CheckPackageHandles for the packages
	// that this file belongs to.
	PackageHandles(ctx context.Context, f File) ([]CheckPackageHandle, error)

	// GetActiveReverseDeps returns the active files belonging to the reverse
	// dependencies of this file's package.
	GetReverseDependencies(id string) []string

	// KnownImportPaths returns all the imported packages loaded in this snapshot,
	// indexed by their import path.
	KnownImportPaths() map[string]Package

	// KnownPackages returns all the packages loaded in this snapshot.
	KnownPackages(ctx context.Context) []Package
}

// CheckPackageHandle represents a handle to a specific version of a package.
// It is uniquely defined by the file handles that make up the package.
type CheckPackageHandle interface {
	// ID returns the ID of the package associated with the CheckPackageHandle.
	ID() string

	// ParseGoHandle returns a ParseGoHandle for which to get the package.
	CompiledGoFiles() []ParseGoHandle

	// Check returns the type-checked Package for the CheckPackageHandle.
	Check(ctx context.Context) (Package, error)

	// Cached returns the Package for the CheckPackageHandle if it has already been stored.
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

	// BuiltinPackage returns the type information for the special "builtin" package.
	BuiltinPackage() BuiltinPackage

	// GetFile returns the file object for a given URI, initializing it
	// if it is not already part of the view.
	GetFile(ctx context.Context, uri span.URI) (File, error)

	// FindFile returns the file object for a given URI if it is
	// already part of the view.
	FindFile(ctx context.Context, uri span.URI) File

	// Called to set the effective contents of a file from this view.
	SetContent(ctx context.Context, uri span.URI, version float64, content []byte)

	// BackgroundContext returns a context used for all background processing
	// on behalf of this view.
	BackgroundContext() context.Context

	// Shutdown closes this view, and detaches it from it's session.
	Shutdown(ctx context.Context)

	// Ignore returns true if this file should be ignored by this view.
	Ignore(span.URI) bool

	// Config returns the configuration for the view.
	Config(ctx context.Context) *packages.Config

	// RunProcessEnvFunc runs fn with the process env for this view inserted into opts.
	// Note: the process env contains cached module and filesystem state.
	RunProcessEnvFunc(ctx context.Context, fn func(*imports.Options) error, opts *imports.Options) error

	// Options returns a copy of the Options for this view.
	Options() Options

	// SetOptions sets the options of this view to new values.
	// Calling this may cause the view to be invalidated and a replacement view
	// added to the session. If so the new view will be returned, otherwise the
	// original one will be.
	SetOptions(context.Context, Options) (View, error)

	// FindFileInPackage returns the AST and type information for a file that may
	// belong to or be part of a dependency of the given package.
	FindPosInPackage(pkg Package, pos token.Pos) (*ast.File, *protocol.ColumnMapper, Package, error)

	// Snapshot returns the current snapshot for the view.
	Snapshot() Snapshot
}

// Session represents a single connection from a client.
// This is the level at which things like open files are maintained on behalf
// of the client.
// A session may have many active views at any given time.
type Session interface {
	// NewView creates a new View and returns it.
	NewView(ctx context.Context, name string, folder span.URI, options Options) (View, []CheckPackageHandle, error)

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

	// DidOpen is invoked each time a file is opened in the editor.
	DidOpen(ctx context.Context, uri span.URI, kind FileKind, version float64, text []byte) error

	// DidSave is invoked each time an open file is saved in the editor.
	DidSave(uri span.URI, version float64)

	// DidClose is invoked each time an open file is closed in the editor.
	DidClose(uri span.URI)

	// IsOpen returns whether the editor currently has a file open.
	IsOpen(uri span.URI) bool

	// Called to set the effective contents of a file from this session.
	SetOverlay(uri span.URI, kind FileKind, version float64, data []byte)

	// DidChangeOutOfBand is called when a file under the root folder changes.
	// If the file was open in the editor, it returns true.
	DidChangeOutOfBand(ctx context.Context, uri span.URI, action FileAction) bool

	// Options returns a copy of the SessionOptions for this session.
	Options() Options

	// SetOptions sets the options of this session to new values.
	SetOptions(Options)
}

type FileAction int

const (
	Open = FileAction(iota)
	Close
	Change
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

	// NewSession creates a new Session manager and returns it.
	NewSession(ctx context.Context) Session

	// FileSet returns the shared fileset used by all files in the system.
	FileSet() *token.FileSet

	// ParseGoHandle returns a ParseGoHandle for the given file handle.
	ParseGoHandle(fh FileHandle, mode ParseMode) ParseGoHandle
}

// FileSystem is the interface to something that provides file contents.
type FileSystem interface {
	// GetFile returns a handle for the specified file.
	GetFile(uri span.URI, kind FileKind) FileHandle
}

// ParseGoHandle represents a handle to the AST for a file.
type ParseGoHandle interface {
	// File returns a file handle for which to get the AST.
	File() FileHandle

	// Mode returns the parse mode of this handle.
	Mode() ParseMode

	// Parse returns the parsed AST for the file.
	// If the file is not available, returns nil and an error.
	Parse(ctx context.Context) (*ast.File, *protocol.ColumnMapper, error, error)

	// Cached returns the AST for this handle, if it has already been stored.
	Cached() (*ast.File, *protocol.ColumnMapper, error, error)
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
	return fmt.Sprintf("%s%f%s%s", fileID.URI, fileID.Version, fileID.Identifier, fileID.Kind)
}

// File represents a source file of any type.
type File interface {
	URI() span.URI
	Kind() FileKind
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

type FileURI span.URI

func (f FileURI) URI() span.URI {
	return span.URI(f)
}

type DirectoryURI span.URI

func (d DirectoryURI) URI() span.URI {
	return span.URI(d)
}

type Scope interface {
	URI() span.URI
}

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
	GetImport(pkgPath string) (Package, error)
	Imports() []Package
}

type Error struct {
	File           FileIdentity
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
	return fmt.Sprintf("%s:%s: %s", e.File, e.Range, e.Message)
}

type BuiltinPackage interface {
	Lookup(name string) *ast.Object
	CompiledGoFiles() []ParseGoHandle
}
