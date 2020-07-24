// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"io"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

// Snapshot represents the current state for the given view.
type Snapshot interface {
	ID() uint64

	// View returns the View associated with this snapshot.
	View() View

	// FindFile returns the FileHandle for the given URI, if it is already
	// in the given snapshot.
	FindFile(uri span.URI) FileHandle

	// GetFile returns the FileHandle for a given URI, initializing it
	// if it is not already part of the snapshot.
	GetFile(ctx context.Context, uri span.URI) (FileHandle, error)

	// IsOpen returns whether the editor currently has a file open.
	IsOpen(uri span.URI) bool

	// IsSaved returns whether the contents are saved on disk or not.
	IsSaved(uri span.URI) bool

	// Analyze runs the analyses for the given package at this snapshot.
	Analyze(ctx context.Context, pkgID string, analyzers ...*analysis.Analyzer) ([]*Error, error)

	// RunGoCommandPiped runs the given `go` command in the view, using the
	// provided stdout and stderr. It will use the -modfile flag, if possible.
	RunGoCommandPiped(ctx context.Context, verb string, args []string, stdout, stderr io.Writer) error

	// RunGoCommand runs the given `go` command in the view. It will use the
	// -modfile flag, if possible.
	RunGoCommand(ctx context.Context, verb string, args []string) (*bytes.Buffer, error)

	// RunGoCommandDirect runs the given `go` command, never using the
	// -modfile flag.
	RunGoCommandDirect(ctx context.Context, verb string, args []string) error

	// ParseModHandle is used to parse go.mod files.
	ParseModHandle(ctx context.Context, fh FileHandle) (ParseModHandle, error)

	// ModWhyHandle is used get the results of `go mod why` for a given module.
	// It only works for go.mod files that can be parsed, hence it takes a
	// ParseModHandle.
	ModWhyHandle(ctx context.Context) (ModWhyHandle, error)

	// ModWhyHandle is used get the possible upgrades for the dependencies of
	// a given module. It only works for go.mod files that can be parsed, hence
	// it takes a ParseModHandle.
	ModUpgradeHandle(ctx context.Context) (ModUpgradeHandle, error)

	// ModWhyHandle is used get the results of `go mod tidy` for a given
	// module. It only works for go.mod files that can be parsed, hence it
	// takes a ParseModHandle.
	ModTidyHandle(ctx context.Context) (ModTidyHandle, error)

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

	// ModFile is the go.mod file at the root of this view. It may not exist.
	ModFile() span.URI

	// BuiltinPackage returns the go/ast.Object for the given name in the builtin package.
	BuiltinPackage(ctx context.Context) (BuiltinPackage, error)

	// BackgroundContext returns a context used for all background processing
	// on behalf of this view.
	BackgroundContext() context.Context

	// Shutdown closes this view, and detaches it from it's session.
	Shutdown(ctx context.Context)

	// WriteEnv writes the view-specific environment to the io.Writer.
	WriteEnv(ctx context.Context, w io.Writer) error

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

	// IsGoPrivatePath reports whether target is a private import path, as identified
	// by the GOPRIVATE environment variable.
	IsGoPrivatePath(path string) bool

	// IgnoredFile reports if a file would be ignored by a `go list` of the whole
	// workspace.
	IgnoredFile(uri span.URI) bool

	// WorkspaceDirectories returns any directory known by the view. For views
	// within a module, this is the module root and any replace targets.
	WorkspaceDirectories(ctx context.Context) ([]string, error)
}

type BuiltinPackage interface {
	Package() *ast.Package
	ParseGoHandle() ParseGoHandle
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

	// GetFile returns a handle for the specified file.
	GetFile(ctx context.Context, uri span.URI) (FileHandle, error)

	// DidModifyFile reports a file modification to the session.
	// It returns the resulting snapshots, a guaranteed one per view.
	DidModifyFiles(ctx context.Context, changes []FileModification) ([]Snapshot, error)

	// Overlays returns a slice of file overlays for the session.
	Overlays() []Overlay

	// Options returns a copy of the SessionOptions for this session.
	Options() Options

	// SetOptions sets the options of this session to new values.
	SetOptions(Options)
}

// Overlay is the type for a file held in memory on a session.
type Overlay interface {
	// Session returns the session this overlay belongs to.
	Session() Session

	// Identity returns the FileIdentity for the overlay.
	Identity() FileIdentity

	// Saved returns whether this overlay has been saved to disk.
	Saved() bool

	// Data is the contents of the overlay held in memory.
	Data() []byte
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

// Cache abstracts the core logic of dealing with the environment from the
// higher level logic that processes the information to produce results.
// The cache provides access to files and their contents, so the source
// package does not directly access the file system.
// A single cache is intended to be process wide, and is the primary point of
// sharing between all consumers.
// A cache may have many active sessions at any given time.
type Cache interface {
	// FileSet returns the shared fileset used by all files in the system.
	FileSet() *token.FileSet

	// GetFile returns a file handle for the given URI.
	GetFile(ctx context.Context, uri span.URI) (FileHandle, error)

	// ParseGoHandle returns a ParseGoHandle for the given file handle.
	ParseGoHandle(ctx context.Context, fh FileHandle, mode ParseMode) ParseGoHandle
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

	// PosToField is a cache of *ast.Fields by token.Pos. This allows us
	// to quickly find corresponding *ast.Field node given a *types.Var.
	// We must refer to the AST to render type aliases properly when
	// formatting signatures and other types.
	PosToField(context.Context) (map[token.Pos]*ast.Field, error)

	// PosToDecl maps certain objects' positions to their surrounding
	// ast.Decl. This mapping is used when building the documentation
	// string for the objects.
	PosToDecl(context.Context) (map[token.Pos]ast.Decl, error)
}

type ParseModHandle interface {
	// Mod returns the file handle for the go.mod file.
	Mod() FileHandle

	// Sum returns the file handle for the analogous go.sum file. It may be nil.
	Sum() FileHandle

	// Parse returns the parsed go.mod file, a column mapper, and a list of
	// parse for the go.mod file.
	Parse(ctx context.Context) (*modfile.File, *protocol.ColumnMapper, []Error, error)
}

type ModUpgradeHandle interface {
	// Upgrades returns the latest versions for each of the module's
	// dependencies.
	Upgrades(ctx context.Context) (map[string]string, error)
}

type ModWhyHandle interface {
	// Why returns the results of `go mod why` for every dependency of the
	// module.
	Why(ctx context.Context) (map[string]string, error)
}

type ModTidyHandle interface {
	// Mod is the ParseModHandle associated with the go.mod file being tidied.
	ParseModHandle() ParseModHandle

	// Tidy returns the results of `go mod tidy` for the module.
	Tidy(ctx context.Context) ([]Error, error)

	// TidiedContent is the content of the tidied go.mod file.
	TidiedContent(ctx context.Context) ([]byte, error)
}

var ErrTmpModfileUnsupported = errors.New("-modfile is unsupported for this Go version")

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

// FileHandle represents a handle to a specific version of a single file.
type FileHandle interface {
	URI() span.URI
	Kind() FileKind
	Version() float64

	// Identity returns a FileIdentity for the file, even if there was an error
	// reading it.
	// It is a fatal error to call Identity on a file that has not yet been read.
	Identity() FileIdentity

	// Read reads the contents of a file.
	// If the file is not available, returns a nil slice and an error.
	Read() ([]byte, error)
}

// FileIdentity uniquely identifies a file at a version from a FileSystem.
type FileIdentity struct {
	URI span.URI

	// SessionID is the ID of the LSP session.
	SessionID string

	// Version is the version of the file, as specified by the client. It should
	// only be set in combination with SessionID.
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
	// UnknownKind is a file type we don't know about.
	UnknownKind = FileKind(iota)

	// Go is a normal go source file.
	Go
	// Mod is a go.mod file.
	Mod
	// Sum is a go.sum file.
	Sum
)

// Analyzer represents a go/analysis analyzer with some boolean properties
// that let the user know how to use the analyzer.
type Analyzer struct {
	Analyzer *analysis.Analyzer
	enabled  bool

	// Command is the name of the command used to invoke the suggested fixes
	// for the analyzer. It is non-nil if we expect this analyzer to provide
	// its fix separately from its diagnostics. That is, we should apply the
	// analyzer's suggested fixes through a Command, not a TextEdit.
	Command *Command

	// If this is true, then we can apply the suggested fixes
	// as part of a source.FixAll codeaction.
	HighConfidence bool

	// FixesError is only set for type-error analyzers.
	// It reports true if the message provided indicates an error that could be
	// fixed by the analyzer.
	FixesError func(msg string) bool
}

func (a Analyzer) Enabled(snapshot Snapshot) bool {
	if enabled, ok := snapshot.View().Options().UserEnabledAnalyses[a.Analyzer.Name]; ok {
		return enabled
	}
	return a.enabled
}

// Package represents a Go package that has been type-checked. It maintains
// only the relevant fields of a *go/packages.Package.
type Package interface {
	ID() string
	Name() string
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
	Module() *packages.Module
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
	ModTidyError
	Analysis
)

func (e *Error) Error() string {
	return fmt.Sprintf("%s:%s: %s", e.URI, e.Range, e.Message)
}

var (
	InconsistentVendoring = errors.New("inconsistent vendoring")
	PackagesLoadError     = errors.New("packages.Load error")
)
