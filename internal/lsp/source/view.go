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

	// Fileset returns the Fileset used to parse all the Go files in this snapshot.
	FileSet() *token.FileSet

	// FindFile returns the FileHandle for the given URI, if it is already
	// in the given snapshot.
	FindFile(uri span.URI) VersionedFileHandle

	// GetFile returns the FileHandle for a given URI, initializing it
	// if it is not already part of the snapshot.
	GetFile(ctx context.Context, uri span.URI) (VersionedFileHandle, error)

	// IsOpen returns whether the editor currently has a file open.
	IsOpen(uri span.URI) bool

	// IsSaved returns whether the contents are saved on disk or not.
	IsSaved(uri span.URI) bool

	// ParseGo returns the parsed AST for the file.
	// If the file is not available, returns nil and an error.
	ParseGo(ctx context.Context, fh FileHandle, mode ParseMode) (*ParsedGoFile, error)

	// PosToField is a cache of *ast.Fields by token.Pos. This allows us
	// to quickly find corresponding *ast.Field node given a *types.Var.
	// We must refer to the AST to render type aliases properly when
	// formatting signatures and other types.
	PosToField(ctx context.Context, pgf *ParsedGoFile) (map[token.Pos]*ast.Field, error)

	// PosToDecl maps certain objects' positions to their surrounding
	// ast.Decl. This mapping is used when building the documentation
	// string for the objects.
	PosToDecl(ctx context.Context, pgf *ParsedGoFile) (map[token.Pos]ast.Decl, error)

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

	// ParseMod is used to parse go.mod files.
	ParseMod(ctx context.Context, fh FileHandle) (*ParsedModule, error)

	// ModWhy returns the results of `go mod why` for the module specified by
	// the given go.mod file.
	ModWhy(ctx context.Context, fh FileHandle) (map[string]string, error)

	// ModUpgrade returns the possible updates for the module specified by the
	// given go.mod file.
	ModUpgrade(ctx context.Context, fh FileHandle) (map[string]string, error)

	// ModTidy returns the results of `go mod tidy` for the module specified by
	// the given go.mod file.
	ModTidy(ctx context.Context, fh FileHandle) (*TidiedModule, error)

	// BuiltinPackage returns information about the special builtin package.
	BuiltinPackage(ctx context.Context) (*BuiltinPackage, error)

	// PackagesForFile returns the packages that this file belongs to, checked
	// in mode.
	PackagesForFile(ctx context.Context, uri span.URI, mode TypecheckMode) ([]Package, error)

	// GetActiveReverseDeps returns the active files belonging to the reverse
	// dependencies of this file's package, checked in TypecheckWorkspace mode.
	GetReverseDependencies(ctx context.Context, id string) ([]Package, error)

	// CachedImportPaths returns all the imported packages loaded in this
	// snapshot, indexed by their import path and checked in TypecheckWorkspace
	// mode.
	CachedImportPaths(ctx context.Context) (map[string]Package, error)

	// KnownPackages returns all the packages loaded in this snapshot, checked
	// in TypecheckWorkspace mode.
	KnownPackages(ctx context.Context) ([]Package, error)

	// WorkspacePackages returns the snapshot's top-level packages.
	WorkspacePackages(ctx context.Context) ([]Package, error)

	// WorkspaceDirectories returns any directory known by the view. For views
	// within a module, this is the module root and any replace targets.
	WorkspaceDirectories(ctx context.Context) []span.URI
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

	// BackgroundContext returns a context used for all background processing
	// on behalf of this view.
	BackgroundContext() context.Context

	// Shutdown closes this view, and detaches it from its session.
	Shutdown(ctx context.Context)

	// AwaitInitialized waits until a view is initialized
	AwaitInitialized(ctx context.Context)

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
	Snapshot(ctx context.Context) (Snapshot, func())

	// Rebuild rebuilds the current view, replacing the original view in its session.
	Rebuild(ctx context.Context) (Snapshot, func(), error)

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
}

type BuiltinPackage struct {
	Package    *ast.Package
	ParsedFile *ParsedGoFile
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
	Mapper   *protocol.ColumnMapper
	ParseErr error
}

// A ParsedModule contains the results of parsing a go.mod file.
type ParsedModule struct {
	File        *modfile.File
	Mapper      *protocol.ColumnMapper
	ParseErrors []Error
}

// A TidiedModule contains the results of running `go mod tidy` on a module.
type TidiedModule struct {
	// The parsed module, which is guaranteed to have parsed successfully.
	Parsed *ParsedModule
	// Diagnostics representing changes made by `go mod tidy`.
	Errors []Error
	// The bytes of the go.mod file after it was tidied.
	TidiedContent []byte
}

// Session represents a single connection from a client.
// This is the level at which things like open files are maintained on behalf
// of the client.
// A session may have many active views at any given time.
type Session interface {
	// NewView creates a new View, returning it and its first snapshot.
	NewView(ctx context.Context, name string, folder span.URI, options Options) (View, Snapshot, func(), error)

	// Cache returns the cache that created this session, for debugging only.
	Cache() interface{}

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

	// DidModifyFile reports a file modification to the session. It returns the
	// resulting snapshots, a guaranteed one per view.
	DidModifyFiles(ctx context.Context, changes []FileModification) ([]Snapshot, []func(), []span.URI, error)

	// Overlays returns a slice of file overlays for the session.
	Overlays() []Overlay

	// Options returns a copy of the SessionOptions for this session.
	Options() Options

	// SetOptions sets the options of this session to new values.
	SetOptions(Options)
}

// Overlay is the type for a file held in memory on a session.
type Overlay interface {
	VersionedFileHandle

	// Saved returns whether this overlay has been saved to disk.
	Saved() bool
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

var ErrTmpModfileUnsupported = errors.New("-modfile is unsupported for this Go version")

// ParseMode controls the content of the AST produced when parsing a source file.
type ParseMode int

const (
	// ParseHeader specifies that the main package declaration and imports are needed.
	// This is the mode used when attempting to examine the package graph structure.
	ParseHeader ParseMode = iota

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

// TypecheckMode controls what kind of parsing should be done (see ParseMode)
// while type checking a package.
type TypecheckMode int

const (
	// Invalid default value.
	TypecheckUnknown TypecheckMode = iota
	// TypecheckFull means to use ParseFull.
	TypecheckFull
	// TypecheckWorkspace means to use ParseFull for workspace packages, and
	// ParseExported for others.
	TypecheckWorkspace
	// TypecheckAll means ParseFull for workspace packages, and both Full and
	// Exported for others. Only valid for some functions.
	TypecheckAll
)

type VersionedFileHandle interface {
	FileHandle
	Version() float64
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
	Version float64
}

// FileHandle represents a handle to a specific version of a single file.
type FileHandle interface {
	URI() span.URI
	Kind() FileKind

	// Identity returns a FileIdentity for the file, even if there was an error
	// reading it.
	// It is a fatal error to call Identity on a file that has not yet been read.
	FileIdentity() FileIdentity
	// Read reads the contents of a file.
	// If the file is not available, returns a nil slice and an error.
	Read() ([]byte, error)
}

// FileIdentity uniquely identifies a file at a version from a FileSystem.
type FileIdentity struct {
	URI span.URI

	// Identifier represents a unique identifier for the file's content.
	Hash string

	// Kind is the file's kind.
	Kind FileKind
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

func (a Analyzer) Enabled(view View) bool {
	if enabled, ok := view.Options().UserEnabledAnalyses[a.Analyzer.Name]; ok {
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
	CompiledGoFiles() []*ParsedGoFile
	File(uri span.URI) (*ParsedGoFile, error)
	GetSyntax() []*ast.File
	GetErrors() []*Error
	GetTypes() *types.Package
	GetTypesInfo() *types.Info
	GetTypesSizes() types.Sizes
	IsIllTyped() bool
	ForTest() string
	GetImport(pkgPath string) (Package, error)
	MissingDependencies() []string
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

// GoModTidy is the source for a diagnostic computed by running `go mod tidy`.
const GoModTidy = "go mod tidy"

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
