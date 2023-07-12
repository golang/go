// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"context"
	"crypto/sha256"
	"encoding/json"
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/scanner"
	"go/token"
	"go/types"
	"io"

	"golang.org/x/mod/modfile"
	"golang.org/x/tools/go/analysis"
	"golang.org/x/tools/go/packages"
	"golang.org/x/tools/go/types/objectpath"
	"golang.org/x/tools/gopls/internal/govulncheck"
	"golang.org/x/tools/gopls/internal/lsp/progress"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/safetoken"
	"golang.org/x/tools/gopls/internal/lsp/source/methodsets"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/event/label"
	"golang.org/x/tools/internal/event/tag"
	"golang.org/x/tools/internal/gocommand"
	"golang.org/x/tools/internal/imports"
	"golang.org/x/tools/internal/packagesinternal"
)

// A GlobalSnapshotID uniquely identifies a snapshot within this process and
// increases monotonically with snapshot creation time.
//
// We use a distinct integral type for global IDs to help enforce correct
// usage.
type GlobalSnapshotID uint64

// Snapshot represents the current state for the given view.
type Snapshot interface {
	// SequenceID is the sequence id of this snapshot within its containing
	// view.
	//
	// Relative to their view sequence ids are monotonically increasing, but this
	// does not hold globally: when new views are created their initial snapshot
	// has sequence ID 0. For operations that span multiple views, use global
	// IDs.
	SequenceID() uint64

	// GlobalID is a globally unique identifier for this snapshot. Global IDs are
	// monotonic: subsequent snapshots will have higher global ID, though
	// subsequent snapshots in a view may not have adjacent global IDs.
	GlobalID() GlobalSnapshotID

	// View returns the View associated with this snapshot.
	View() View

	// BackgroundContext returns a context used for all background processing
	// on behalf of this snapshot.
	BackgroundContext() context.Context

	// A Snapshot is a caching implementation of FileSource whose
	// ReadFile method returns consistent information about the existence
	// and content of each file throughout its lifetime.
	FileSource

	// FindFile returns the FileHandle for the given URI, if it is already
	// in the given snapshot.
	// TODO(adonovan): delete this operation; use ReadFile instead.
	FindFile(uri span.URI) FileHandle

	// AwaitInitialized waits until the snapshot's view is initialized.
	AwaitInitialized(ctx context.Context)

	// IsOpen returns whether the editor currently has a file open.
	IsOpen(uri span.URI) bool

	// IgnoredFile reports if a file would be ignored by a `go list` of the whole
	// workspace.
	IgnoredFile(uri span.URI) bool

	// Templates returns the .tmpl files
	Templates() map[span.URI]FileHandle

	// ParseGo returns the parsed AST for the file.
	// If the file is not available, returns nil and an error.
	// Position information is added to FileSet().
	ParseGo(ctx context.Context, fh FileHandle, mode parser.Mode) (*ParsedGoFile, error)

	// Analyze runs the specified analyzers on the given packages at this snapshot.
	//
	// If the provided tracker is non-nil, it may be used to report progress of
	// the analysis pass.
	Analyze(ctx context.Context, pkgIDs map[PackageID]unit, analyzers []*Analyzer, tracker *progress.Tracker) ([]*Diagnostic, error)

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
	RunProcessEnvFunc(ctx context.Context, fn func(context.Context, *imports.Options) error) error

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

	// ModVuln returns import vulnerability analysis for the given go.mod URI.
	// Concurrent requests are combined into a single command.
	ModVuln(ctx context.Context, modURI span.URI) (*govulncheck.Result, error)

	// GoModForFile returns the URI of the go.mod file for the given URI.
	GoModForFile(uri span.URI) span.URI

	// WorkFile, if non-empty, is the go.work file for the workspace.
	WorkFile() span.URI

	// ParseWork is used to parse go.work files.
	ParseWork(ctx context.Context, fh FileHandle) (*ParsedWorkFile, error)

	// BuiltinFile returns information about the special builtin package.
	BuiltinFile(ctx context.Context) (*ParsedGoFile, error)

	// IsBuiltin reports whether uri is part of the builtin package.
	IsBuiltin(uri span.URI) bool

	// CriticalError returns any critical errors in the workspace.
	//
	// A nil result may mean success, or context cancellation.
	CriticalError(ctx context.Context) *CriticalError

	// Symbols returns all symbols in the snapshot.
	//
	// If workspaceOnly is set, this only includes symbols from files in a
	// workspace package. Otherwise, it returns symbols from all loaded packages.
	Symbols(ctx context.Context, workspaceOnly bool) (map[span.URI][]Symbol, error)

	// -- package metadata --

	// ReverseDependencies returns a new mapping whose entries are
	// the ID and Metadata of each package in the workspace that
	// directly or transitively depend on the package denoted by id,
	// excluding id itself.
	ReverseDependencies(ctx context.Context, id PackageID, transitive bool) (map[PackageID]*Metadata, error)

	// WorkspaceMetadata returns a new, unordered slice containing
	// metadata for all ordinary and test packages (but not
	// intermediate test variants) in the workspace.
	//
	// The workspace is the set of modules typically defined by a
	// go.work file. It is not transitively closed: for example,
	// the standard library is not usually part of the workspace
	// even though every module in the workspace depends on it.
	//
	// Operations that must inspect all the dependencies of the
	// workspace packages should instead use AllMetadata.
	WorkspaceMetadata(ctx context.Context) ([]*Metadata, error)

	// AllMetadata returns a new unordered array of metadata for
	// all packages known to this snapshot, which includes the
	// packages of all workspace modules plus their transitive
	// import dependencies.
	//
	// It may also contain ad-hoc packages for standalone files.
	// It includes all test variants.
	AllMetadata(ctx context.Context) ([]*Metadata, error)

	// Metadata returns the metadata for the specified package,
	// or nil if it was not found.
	Metadata(id PackageID) *Metadata

	// MetadataForFile returns a new slice containing metadata for each
	// package containing the Go file identified by uri, ordered by the
	// number of CompiledGoFiles (i.e. "narrowest" to "widest" package),
	// and secondarily by IsIntermediateTestVariant (false < true).
	// The result may include tests and intermediate test variants of
	// importable packages.
	// It returns an error if the context was cancelled.
	MetadataForFile(ctx context.Context, uri span.URI) ([]*Metadata, error)

	// OrphanedFileDiagnostics reports diagnostics for files that have no package
	// associations or which only have only command-line-arguments packages.
	//
	// The caller must not mutate the result.
	OrphanedFileDiagnostics(ctx context.Context) (map[span.URI]*Diagnostic, error)

	// -- package type-checking --

	// TypeCheck parses and type-checks the specified packages,
	// and returns them in the same order as the ids.
	// The resulting packages' types may belong to different importers,
	// so types from different packages are incommensurable.
	//
	// In general, clients should never need to type-checked
	// syntax for an intermediate test variant (ITV) package.
	// Callers should apply RemoveIntermediateTestVariants (or
	// equivalent) before this method, or any of the potentially
	// type-checking methods below.
	TypeCheck(ctx context.Context, ids ...PackageID) ([]Package, error)

	// PackageDiagnostics returns diagnostics for files contained in specified
	// packages.
	//
	// If these diagnostics cannot be loaded from cache, the requested packages
	// may be type-checked.
	PackageDiagnostics(ctx context.Context, ids ...PackageID) (map[span.URI][]*Diagnostic, error)

	// References returns cross-references indexes for the specified packages.
	//
	// If these indexes cannot be loaded from cache, the requested packages may
	// be type-checked.
	References(ctx context.Context, ids ...PackageID) ([]XrefIndex, error)

	// MethodSets returns method-set indexes for the specified packages.
	//
	// If these indexes cannot be loaded from cache, the requested packages may
	// be type-checked.
	MethodSets(ctx context.Context, ids ...PackageID) ([]*methodsets.Index, error)
}

// NarrowestMetadataForFile returns metadata for the narrowest package
// (the one with the fewest files) that encloses the specified file.
// The result may be a test variant, but never an intermediate test variant.
func NarrowestMetadataForFile(ctx context.Context, snapshot Snapshot, uri span.URI) (*Metadata, error) {
	metas, err := snapshot.MetadataForFile(ctx, uri)
	if err != nil {
		return nil, err
	}
	RemoveIntermediateTestVariants(&metas)
	if len(metas) == 0 {
		return nil, fmt.Errorf("no package metadata for file %s", uri)
	}
	return metas[0], nil
}

type XrefIndex interface {
	Lookup(targets map[PackagePath]map[objectpath.Path]struct{}) (locs []protocol.Location)
}

// SnapshotLabels returns a new slice of labels that should be used for events
// related to a snapshot.
func SnapshotLabels(snapshot Snapshot) []label.Label {
	return []label.Label{tag.Snapshot.Of(snapshot.SequenceID()), tag.Directory.Of(snapshot.View().Folder())}
}

// NarrowestPackageForFile is a convenience function that selects the
// narrowest non-ITV package to which this file belongs, type-checks
// it in the requested mode (full or workspace), and returns it, along
// with the parse tree of that file.
//
// The "narrowest" package is the one with the fewest number of files
// that includes the given file. This solves the problem of test
// variants, as the test will have more files than the non-test package.
// (Historically the preference was a parameter but widest was almost
// never needed.)
//
// An intermediate test variant (ITV) package has identical source
// to a regular package but resolves imports differently.
// gopls should never need to type-check them.
//
// Type-checking is expensive. Call snapshot.ParseGo if all you need
// is a parse tree, or snapshot.MetadataForFile if you only need metadata.
func NarrowestPackageForFile(ctx context.Context, snapshot Snapshot, uri span.URI) (Package, *ParsedGoFile, error) {
	metas, err := snapshot.MetadataForFile(ctx, uri)
	if err != nil {
		return nil, nil, err
	}
	RemoveIntermediateTestVariants(&metas)
	if len(metas) == 0 {
		return nil, nil, fmt.Errorf("no package metadata for file %s", uri)
	}
	narrowest := metas[0]
	pkgs, err := snapshot.TypeCheck(ctx, narrowest.ID)
	if err != nil {
		return nil, nil, err
	}
	pkg := pkgs[0]
	pgf, err := pkg.File(uri)
	if err != nil {
		return nil, nil, err // "can't happen"
	}
	return pkg, pgf, err
}

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
	// ID returns a globally unique identifier for this view.
	ID() string

	// Name returns the name this view was constructed with.
	Name() string

	// Folder returns the folder with which this view was created.
	Folder() span.URI

	// Options returns a copy of the Options for this view.
	Options() *Options

	// Snapshot returns the current snapshot for the view, and a
	// release function that must be called when the Snapshot is
	// no longer needed.
	//
	// If the view is shut down, the resulting error will be non-nil, and the
	// release function need not be called.
	Snapshot() (Snapshot, func(), error)

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

	// Vulnerabilities returns known vulnerabilities for the given modfile.
	// TODO(suzmue): replace command.Vuln with a different type, maybe
	// https://pkg.go.dev/golang.org/x/vuln/cmd/govulncheck/govulnchecklib#Summary?
	Vulnerabilities(modfile ...span.URI) map[span.URI]*govulncheck.Result

	// SetVulnerabilities resets the list of vulnerabilities that exists for the given modules
	// required by modfile.
	SetVulnerabilities(modfile span.URI, vulncheckResult *govulncheck.Result)

	// FileKind returns the type of a file.
	//
	// We can't reliably deduce the kind from the file name alone,
	// as some editors can be told to interpret a buffer as
	// language different from the file name heuristic, e.g. that
	// an .html file actually contains Go "html/template" syntax,
	// or even that a .go file contains Python.
	FileKind(FileHandle) FileKind

	// GoVersion returns the configured Go version for this view.
	GoVersion() int

	// GoVersionString returns the go version string configured for this view.
	// Unlike [GoVersion], this encodes the minor version and commit hash information.
	GoVersionString() string
}

// A FileSource maps URIs to FileHandles.
type FileSource interface {
	// ReadFile returns the FileHandle for a given URI, either by
	// reading the content of the file or by obtaining it from a cache.
	ReadFile(ctx context.Context, uri span.URI) (FileHandle, error)
}

// A MetadataSource maps package IDs to metadata.
//
// TODO(rfindley): replace this with a concrete metadata graph, once it is
// exposed from the snapshot.
type MetadataSource interface {
	// Metadata returns Metadata for the given package ID, or nil if it does not
	// exist.
	Metadata(PackageID) *Metadata
}

// A ParsedGoFile contains the results of parsing a Go file.
type ParsedGoFile struct {
	URI  span.URI
	Mode parser.Mode
	File *ast.File
	Tok  *token.File
	// Source code used to build the AST. It may be different from the
	// actual content of the file if we have fixed the AST.
	Src []byte

	// FixedSrc and Fixed AST report on "fixing" that occurred during parsing of
	// this file.
	//
	// If FixedSrc == true, the source contained in the Src field was modified
	// from the original source to improve parsing.
	//
	// If FixedAST == true, the ast was modified after parsing, and therefore
	// positions encoded in the AST may not accurately represent the content of
	// the Src field.
	//
	// TODO(rfindley): there are many places where we haphazardly use the Src or
	// positions without checking these fields. Audit these places and guard
	// accordingly. After doing so, we may find that we don't need to
	// differentiate FixedSrc and FixedAST.
	FixedSrc bool
	FixedAST bool
	Mapper   *protocol.Mapper // may map fixed Src, not file content
	ParseErr scanner.ErrorList
}

// Fixed reports whether p was "Fixed", meaning that its source or positions
// may not correlate with the original file.
func (p ParsedGoFile) Fixed() bool {
	return p.FixedSrc || p.FixedAST
}

// -- go/token domain convenience helpers --

// PositionPos returns the token.Pos of protocol position p within the file.
func (pgf *ParsedGoFile) PositionPos(p protocol.Position) (token.Pos, error) {
	offset, err := pgf.Mapper.PositionOffset(p)
	if err != nil {
		return token.NoPos, err
	}
	return safetoken.Pos(pgf.Tok, offset)
}

// PosRange returns a protocol Range for the token.Pos interval in this file.
func (pgf *ParsedGoFile) PosRange(start, end token.Pos) (protocol.Range, error) {
	return pgf.Mapper.PosRange(pgf.Tok, start, end)
}

// PosMappedRange returns a MappedRange for the token.Pos interval in this file.
// A MappedRange can be converted to any other form.
func (pgf *ParsedGoFile) PosMappedRange(start, end token.Pos) (protocol.MappedRange, error) {
	return pgf.Mapper.PosMappedRange(pgf.Tok, start, end)
}

// PosLocation returns a protocol Location for the token.Pos interval in this file.
func (pgf *ParsedGoFile) PosLocation(start, end token.Pos) (protocol.Location, error) {
	return pgf.Mapper.PosLocation(pgf.Tok, start, end)
}

// NodeRange returns a protocol Range for the ast.Node interval in this file.
func (pgf *ParsedGoFile) NodeRange(node ast.Node) (protocol.Range, error) {
	return pgf.Mapper.NodeRange(pgf.Tok, node)
}

// NodeMappedRange returns a MappedRange for the ast.Node interval in this file.
// A MappedRange can be converted to any other form.
func (pgf *ParsedGoFile) NodeMappedRange(node ast.Node) (protocol.MappedRange, error) {
	return pgf.Mapper.NodeMappedRange(pgf.Tok, node)
}

// NodeLocation returns a protocol Location for the ast.Node interval in this file.
func (pgf *ParsedGoFile) NodeLocation(node ast.Node) (protocol.Location, error) {
	return pgf.Mapper.PosLocation(pgf.Tok, node.Pos(), node.End())
}

// RangePos parses a protocol Range back into the go/token domain.
func (pgf *ParsedGoFile) RangePos(r protocol.Range) (token.Pos, token.Pos, error) {
	start, end, err := pgf.Mapper.RangeOffsets(r)
	if err != nil {
		return token.NoPos, token.NoPos, err
	}
	return pgf.Tok.Pos(start), pgf.Tok.Pos(end), nil
}

// A ParsedModule contains the results of parsing a go.mod file.
type ParsedModule struct {
	URI         span.URI
	File        *modfile.File
	Mapper      *protocol.Mapper
	ParseErrors []*Diagnostic
}

// A ParsedWorkFile contains the results of parsing a go.work file.
type ParsedWorkFile struct {
	URI         span.URI
	File        *modfile.WorkFile
	Mapper      *protocol.Mapper
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
// The Deps* maps do not contain self-import edges.
//
// An ad-hoc package (without go.mod or GOPATH) has its ID, PkgPath,
// and LoadDir equal to the absolute path of its directory.
type Metadata struct {
	ID      PackageID
	PkgPath PackagePath
	Name    PackageName

	// these three fields are as defined by go/packages.Package
	GoFiles         []span.URI
	CompiledGoFiles []span.URI
	IgnoredFiles    []span.URI

	ForTest       PackagePath // q in a "p [q.test]" package, else ""
	TypesSizes    types.Sizes
	Errors        []packages.Error          // must be set for packages in import cycles
	DepsByImpPath map[ImportPath]PackageID  // may contain dups; empty ID => missing
	DepsByPkgPath map[PackagePath]PackageID // values are unique and non-empty
	Module        *packages.Module
	DepsErrors    []*packagesinternal.PackageError
	Diagnostics   []*Diagnostic // processed diagnostics from 'go list'
	LoadDir       string        // directory from which go/packages was run
	Standalone    bool          // package synthesized for a standalone file (e.g. ignore-tagged)
}

func (m *Metadata) String() string { return string(m.ID) }

// IsIntermediateTestVariant reports whether the given package is an
// intermediate test variant (ITV), e.g. "net/http [net/url.test]".
//
// An ITV has identical syntax to the regular variant, but different
// import metadata (DepsBy{Imp,Pkg}Path).
//
// Such test variants arise when an x_test package (in this case net/url_test)
// imports a package (in this case net/http) that itself imports the
// non-x_test package (in this case net/url).
//
// This is done so that the forward transitive closure of net/url_test has
// only one package for the "net/url" import.
// The ITV exists to hold the test variant import:
//
// net/url_test [net/url.test]
//
//	| "net/http" -> net/http [net/url.test]
//	| "net/url" -> net/url [net/url.test]
//	| ...
//
// net/http [net/url.test]
//
//	| "net/url" -> net/url [net/url.test]
//	| ...
//
// This restriction propagates throughout the import graph of net/http: for
// every package imported by net/http that imports net/url, there must be an
// intermediate test variant that instead imports "net/url [net/url.test]".
//
// As one can see from the example of net/url and net/http, intermediate test
// variants can result in many additional packages that are essentially (but
// not quite) identical. For this reason, we filter these variants wherever
// possible.
//
// # Why we mostly ignore intermediate test variants
//
// In projects with complicated tests, there may be a very large
// number of ITVs--asymptotically more than the number of ordinary
// variants. Since they have identical syntax, it is fine in most
// cases to ignore them since the results of analyzing the ordinary
// variant suffice. However, this is not entirely sound.
//
// Consider this package:
//
//	// p/p.go -- in all variants of p
//	package p
//	type T struct { io.Closer }
//
//	// p/p_test.go -- in test variant of p
//	package p
//	func (T) Close() error { ... }
//
// The ordinary variant "p" defines T with a Close method promoted
// from io.Closer. But its test variant "p [p.test]" defines a type T
// with a Close method from p_test.go.
//
// Now consider a package q that imports p, perhaps indirectly. Within
// it, T.Close will resolve to the first Close method:
//
//	// q/q.go -- in all variants of q
//	package q
//	import "p"
//	var _ = new(p.T).Close
//
// Let's assume p also contains this file defining an external test (xtest):
//
//	// p/p_x_test.go -- external test of p
//	package p_test
//	import ( "q"; "testing" )
//	func Test(t *testing.T) { ... }
//
// Note that q imports p, but p's xtest imports q. Now, in "q
// [p.test]", the intermediate test variant of q built for p's
// external test, T.Close resolves not to the io.Closer.Close
// interface method, but to the concrete method of T.Close
// declared in p_test.go.
//
// If we now request all references to the T.Close declaration in
// p_test.go, the result should include the reference from q's ITV.
// (It's not just methods that can be affected; fields can too, though
// it requires bizarre code to achieve.)
//
// As a matter of policy, gopls mostly ignores this subtlety,
// because to account for it would require that we type-check every
// intermediate test variant of p, of which there could be many.
// Good code doesn't rely on such trickery.
//
// Most callers of MetadataForFile call RemoveIntermediateTestVariants
// to discard them before requesting type checking, or the products of
// type-checking such as the cross-reference index or method set index.
//
// MetadataForFile doesn't do this filtering itself becaused in some
// cases we need to make a reverse dependency query on the metadata
// graph, and it's important to include the rdeps of ITVs in that
// query. But the filtering of ITVs should be applied after that step,
// before type checking.
//
// In general, we should never type check an ITV.
func (m *Metadata) IsIntermediateTestVariant() bool {
	return m.ForTest != "" && m.ForTest != m.PkgPath && m.ForTest+"_test" != m.PkgPath
}

// RemoveIntermediateTestVariants removes intermediate test variants, modifying the array.
// We use a pointer to a slice make it impossible to forget to use the result.
func RemoveIntermediateTestVariants(pmetas *[]*Metadata) {
	metas := *pmetas
	res := metas[:0]
	for _, m := range metas {
		if !m.IsIntermediateTestVariant() {
			res = append(res, m)
		}
	}
	*pmetas = res
}

var ErrViewExists = errors.New("view already exists for session")

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

// Common parse modes; these should be reused wherever possible to increase
// cache hits.
const (
	// ParseHeader specifies that the main package declaration and imports are needed.
	// This is the mode used when attempting to examine the package graph structure.
	ParseHeader = parser.AllErrors | parser.ParseComments | parser.ImportsOnly | SkipObjectResolution

	// ParseFull specifies the full AST is needed.
	// This is used for files of direct interest where the entire contents must
	// be considered.
	ParseFull = parser.AllErrors | parser.ParseComments | SkipObjectResolution
)

// A FileHandle represents the URI, content, hash, and optional
// version of a file tracked by the LSP session.
//
// File content may be provided by the file system (for Saved files)
// or from an overlay, for open files with unsaved edits.
// A FileHandle may record an attempt to read a non-existent file,
// in which case Content returns an error.
type FileHandle interface {
	// URI is the URI for this file handle.
	// TODO(rfindley): this is not actually well-defined. In some cases, there
	// may be more than one URI that resolve to the same FileHandle. Which one is
	// this?
	URI() span.URI
	// FileIdentity returns a FileIdentity for the file, even if there was an
	// error reading it.
	FileIdentity() FileIdentity
	// Saved reports whether the file has the same content on disk:
	// it is false for files open on an editor with unsaved edits.
	Saved() bool
	// Version returns the file version, as defined by the LSP client.
	// For on-disk file handles, Version returns 0.
	Version() int32
	// Content returns the contents of a file.
	// If the file is not available, returns a nil slice and an error.
	Content() ([]byte, error)
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

func (k FileKind) String() string {
	switch k {
	case Go:
		return "go"
	case Mod:
		return "go.mod"
	case Sum:
		return "go.sum"
	case Tmpl:
		return "tmpl"
	case Work:
		return "go.work"
	default:
		return fmt.Sprintf("internal error: unknown file kind %d", k)
	}
}

// Analyzer represents a go/analysis analyzer with some boolean properties
// that let the user know how to use the analyzer.
type Analyzer struct {
	Analyzer *analysis.Analyzer

	// Enabled reports whether the analyzer is enabled. This value can be
	// configured per-analysis in user settings. For staticcheck analyzers,
	// the value of the Staticcheck setting overrides this field.
	//
	// Most clients should use the IsEnabled method.
	Enabled bool

	// Fix is the name of the suggested fix name used to invoke the suggested
	// fixes for the analyzer. It is non-empty if we expect this analyzer to
	// provide its fix separately from its diagnostics. That is, we should apply
	// the analyzer's suggested fixes through a Command, not a TextEdit.
	Fix string

	// fixesDiagnostic reports if a diagnostic from the analyzer can be fixed by Fix.
	// If nil then all diagnostics from the analyzer are assumed to be fixable.
	fixesDiagnostic func(*Diagnostic) bool

	// ActionKind is the kind of code action this analyzer produces. If
	// unspecified the type defaults to quickfix.
	ActionKind []protocol.CodeActionKind

	// Severity is the severity set for diagnostics reported by this
	// analyzer. If left unset it defaults to Warning.
	Severity protocol.DiagnosticSeverity

	// Tag is extra tags (unnecessary, deprecated, etc) for diagnostics
	// reported by this analyzer.
	Tag []protocol.DiagnosticTag
}

func (a *Analyzer) String() string { return a.Analyzer.String() }

// IsEnabled reports whether this analyzer is enabled by the given options.
func (a Analyzer) IsEnabled(options *Options) bool {
	// Staticcheck analyzers can only be enabled when staticcheck is on.
	if _, ok := options.StaticcheckAnalyzers[a.Analyzer.Name]; ok {
		if !options.Staticcheck {
			return false
		}
	}
	if enabled, ok := options.Analyses[a.Analyzer.Name]; ok {
		return enabled
	}
	return a.Enabled
}

// FixesDiagnostic returns true if Analyzer.Fix can fix the Diagnostic.
func (a Analyzer) FixesDiagnostic(d *Diagnostic) bool {
	if a.fixesDiagnostic == nil {
		return true
	}
	return a.fixesDiagnostic(d)
}

// Declare explicit types for package paths, names, and IDs to ensure that we
// never use an ID where a path belongs, and vice versa. If we confused these,
// it would result in confusing errors because package IDs often look like
// package paths.
type (
	PackageID   string // go list's unique identifier for a package (e.g. "vendor/example.com/foo [vendor/example.com/bar.test]")
	PackagePath string // name used to prefix linker symbols (e.g. "vendor/example.com/foo")
	PackageName string // identifier in 'package' declaration (e.g. "foo")
	ImportPath  string // path that appears in an import declaration (e.g. "example.com/foo")
)

// Package represents a Go package that has been parsed and type-checked.
//
// By design, there is no way to reach from a Package to the Package
// representing one of its dependencies.
//
// Callers must not assume that two Packages share the same
// token.FileSet or types.Importer and thus have commensurable
// token.Pos values or types.Objects. Instead, use stable naming
// schemes, such as (URI, byte offset) for positions, or (PackagePath,
// objectpath.Path) for exported declarations.
type Package interface {
	Metadata() *Metadata

	// Results of parsing:
	FileSet() *token.FileSet
	CompiledGoFiles() []*ParsedGoFile // (borrowed)
	File(uri span.URI) (*ParsedGoFile, error)
	GetSyntax() []*ast.File // (borrowed)
	GetParseErrors() []scanner.ErrorList

	// Results of type checking:
	GetTypes() *types.Package
	GetTypeErrors() []types.Error
	GetTypesInfo() *types.Info
	DependencyTypes(PackagePath) *types.Package // nil for indirect dependency of no consequence
	DiagnosticsForFile(ctx context.Context, s Snapshot, uri span.URI) ([]*Diagnostic, error)
}

type unit = struct{}

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
	// TODO(adonovan): should be a protocol.URI, for symmetry.
	URI      span.URI // of diagnosed file (not diagnostic documentation)
	Range    protocol.Range
	Severity protocol.DiagnosticSeverity
	Code     string
	CodeHref string

	// Source is a human-readable description of the source of the error.
	// Diagnostics generated by an analysis.Analyzer set it to Analyzer.Name.
	Source DiagnosticSource

	Message string

	Tags    []protocol.DiagnosticTag
	Related []protocol.DiagnosticRelatedInformation

	// Fields below are used internally to generate quick fixes. They aren't
	// part of the LSP spec and historically didn't leave the server.
	//
	// Update(2023-05): version 3.16 of the LSP spec included support for the
	// Diagnostic.data field, which holds arbitrary data preserved in the
	// diagnostic for codeAction requests. This field allows bundling additional
	// information for quick-fixes, and gopls can (and should) use this
	// information to avoid re-evaluating diagnostics in code-action handlers.
	//
	// In order to stage this transition incrementally, the 'BundledFixes' field
	// may store a 'bundled' (=json-serialized) form of the associated
	// SuggestedFixes. Not all diagnostics have their fixes bundled.
	BundledFixes   *json.RawMessage
	SuggestedFixes []SuggestedFix
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
	Vulncheck                DiagnosticSource = "vulncheck imports"
	Govulncheck              DiagnosticSource = "govulncheck"
	TemplateError            DiagnosticSource = "template"
	WorkFileError            DiagnosticSource = "go.work file"
	ConsistencyInfo          DiagnosticSource = "consistency"
)

func AnalyzerErrorKind(name string) DiagnosticSource {
	return DiagnosticSource(name)
}
