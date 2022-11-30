// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/printer"
	"go/token"
	"go/types"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/bug"
	"golang.org/x/tools/internal/typeparams"
)

// MappedRange provides mapped protocol.Range for a span.Range, accounting for
// UTF-16 code points.
//
// TOOD(adonovan): stop treating //line directives specially, and
// eliminate this type. All callers need either m, or a protocol.Range.
type MappedRange struct {
	spanRange span.Range             // the range in the compiled source (package.CompiledGoFiles)
	m         *protocol.ColumnMapper // a mapper of the edited source (package.GoFiles)
}

// NewMappedRange returns a MappedRange for the given file and valid start/end token.Pos.
//
// By convention, start and end are assumed to be positions in the compiled (==
// type checked) source, whereas the column mapper m maps positions in the
// user-edited source. Note that these may not be the same, as when using goyacc or CGo:
// CompiledGoFiles contains generated files, whose positions (via
// token.File.Position) point to locations in the edited file -- the file
// containing `import "C"`.
func NewMappedRange(file *token.File, m *protocol.ColumnMapper, start, end token.Pos) MappedRange {
	mapped := m.TokFile.Name()
	adjusted := file.PositionFor(start, true) // adjusted position
	if adjusted.Filename != mapped {
		bug.Reportf("mapped file %q does not match start position file %q", mapped, adjusted.Filename)
	}
	return MappedRange{
		spanRange: span.NewRange(file, start, end),
		m:         m,
	}
}

// Range returns the LSP range in the edited source.
//
// See the documentation of NewMappedRange for information on edited vs
// compiled source.
func (s MappedRange) Range() (protocol.Range, error) {
	if s.m == nil {
		return protocol.Range{}, bug.Errorf("invalid range")
	}
	spn, err := s.Span()
	if err != nil {
		return protocol.Range{}, err
	}
	return s.m.Range(spn)
}

// Span returns the span corresponding to the mapped range in the edited
// source.
//
// See the documentation of NewMappedRange for information on edited vs
// compiled source.
func (s MappedRange) Span() (span.Span, error) {
	// In the past, some code-paths have relied on Span returning an error if s
	// is the zero value (i.e. s.m is nil). But this should be treated as a bug:
	// observe that s.URI() would panic in this case.
	if s.m == nil {
		return span.Span{}, bug.Errorf("invalid range")
	}
	return span.FileSpan(s.spanRange.TokFile, s.m.TokFile, s.spanRange.Start, s.spanRange.End)
}

// URI returns the URI of the edited file.
//
// See the documentation of NewMappedRange for information on edited vs
// compiled source.
func (s MappedRange) URI() span.URI {
	return s.m.URI
}

// GetParsedFile is a convenience function that extracts the Package and
// ParsedGoFile for a file in a Snapshot. pkgPolicy is one of NarrowestPackage/
// WidestPackage.
//
// Type-checking is expensive. Call cheaper methods of Snapshot if all
// you need is Metadata or parse trees.
func GetParsedFile(ctx context.Context, snapshot Snapshot, fh FileHandle, pkgPolicy PackageFilter) (Package, *ParsedGoFile, error) {
	pkg, err := snapshot.PackageForFile(ctx, fh.URI(), TypecheckWorkspace, pkgPolicy)
	if err != nil {
		return nil, nil, err
	}
	pgh, err := pkg.File(fh.URI())
	return pkg, pgh, err
}

func IsGenerated(ctx context.Context, snapshot Snapshot, uri span.URI) bool {
	fh, err := snapshot.GetFile(ctx, uri)
	if err != nil {
		return false
	}
	pgf, err := snapshot.ParseGo(ctx, fh, ParseHeader)
	if err != nil {
		return false
	}
	for _, commentGroup := range pgf.File.Comments {
		for _, comment := range commentGroup.List {
			if matched := generatedRx.MatchString(comment.Text); matched {
				// Check if comment is at the beginning of the line in source.
				if pgf.Tok.Position(comment.Slash).Column == 1 {
					return true
				}
			}
		}
	}
	return false
}

func objToMappedRange(pkg Package, obj types.Object) (MappedRange, error) {
	nameLen := len(obj.Name())
	if pkgName, ok := obj.(*types.PkgName); ok {
		// An imported Go package has a package-local, unqualified name.
		// When the name matches the imported package name, there is no
		// identifier in the import spec with the local package name.
		//
		// For example:
		// 		import "go/ast" 	// name "ast" matches package name
		// 		import a "go/ast"  	// name "a" does not match package name
		//
		// When the identifier does not appear in the source, have the range
		// of the object be the import path, including quotes.
		if pkgName.Imported().Name() == pkgName.Name() {
			nameLen = len(pkgName.Imported().Path()) + len(`""`)
		}
	}
	return posToMappedRange(pkg, obj.Pos(), obj.Pos()+token.Pos(nameLen))
}

// posToMappedRange returns the MappedRange for the given [start, end) span,
// which must be among the transitive dependencies of pkg.
func posToMappedRange(pkg Package, pos, end token.Pos) (MappedRange, error) {
	if !pos.IsValid() {
		return MappedRange{}, fmt.Errorf("invalid start position")
	}
	if !end.IsValid() {
		return MappedRange{}, fmt.Errorf("invalid end position")
	}

	fset := pkg.FileSet()
	tokFile := fset.File(pos)
	// Subtle: it is not safe to simplify this to tokFile.Name
	// because, due to //line directives, a Position within a
	// token.File may have a different filename than the File itself.
	logicalFilename := tokFile.Position(pos).Filename
	pgf, _, err := findFileInDeps(pkg, span.URIFromPath(logicalFilename))
	if err != nil {
		return MappedRange{}, err
	}
	// It is problematic that pgf.Mapper (from the parsed Go file) is
	// accompanied here not by pgf.Tok but by tokFile from the global
	// FileSet, which is a distinct token.File that doesn't
	// contain [pos,end).
	//
	// This is done because tokFile is the *token.File for the compiled go file
	// containing pos, whereas Mapper is the UTF16 mapper for the go file pointed
	// to by line directives.
	//
	// TODO(golang/go#55043): clean this up.
	return NewMappedRange(tokFile, pgf.Mapper, pos, end), nil
}

// FindPackageFromPos returns the Package for the given position, which must be
// among the transitive dependencies of pkg.
//
// TODO(rfindley): is this the best factoring of this API? This function is
// really a trivial wrapper around findFileInDeps, which may be a more useful
// function to expose.
func FindPackageFromPos(pkg Package, pos token.Pos) (Package, error) {
	if !pos.IsValid() {
		return nil, fmt.Errorf("invalid position")
	}
	fileName := pkg.FileSet().File(pos).Name()
	uri := span.URIFromPath(fileName)
	_, pkg, err := findFileInDeps(pkg, uri)
	return pkg, err
}

// Matches cgo generated comment as well as the proposed standard:
//
//	https://golang.org/s/generatedcode
var generatedRx = regexp.MustCompile(`// .*DO NOT EDIT\.?`)

// FileKindForLang returns the file kind associated with the given language ID,
// or UnknownKind if the language ID is not recognized.
func FileKindForLang(langID string) FileKind {
	switch langID {
	case "go":
		return Go
	case "go.mod":
		return Mod
	case "go.sum":
		return Sum
	case "tmpl", "gotmpl":
		return Tmpl
	case "go.work":
		return Work
	default:
		return UnknownKind
	}
}

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
		return fmt.Sprintf("unk%d", k)
	}
}

// nodeAtPos returns the index and the node whose position is contained inside
// the node list.
func nodeAtPos(nodes []ast.Node, pos token.Pos) (ast.Node, int) {
	if nodes == nil {
		return nil, -1
	}
	for i, node := range nodes {
		if node.Pos() <= pos && pos <= node.End() {
			return node, i
		}
	}
	return nil, -1
}

// IsInterface returns if a types.Type is an interface
func IsInterface(T types.Type) bool {
	return T != nil && types.IsInterface(T)
}

// FormatNode returns the "pretty-print" output for an ast node.
func FormatNode(fset *token.FileSet, n ast.Node) string {
	var buf strings.Builder
	if err := printer.Fprint(&buf, fset, n); err != nil {
		return ""
	}
	return buf.String()
}

// Deref returns a pointer's element type, traversing as many levels as needed.
// Otherwise it returns typ.
//
// It can return a pointer type for cyclic types (see golang/go#45510).
func Deref(typ types.Type) types.Type {
	var seen map[types.Type]struct{}
	for {
		p, ok := typ.Underlying().(*types.Pointer)
		if !ok {
			return typ
		}
		if _, ok := seen[p.Elem()]; ok {
			return typ
		}

		typ = p.Elem()

		if seen == nil {
			seen = make(map[types.Type]struct{})
		}
		seen[typ] = struct{}{}
	}
}

func SortDiagnostics(d []*Diagnostic) {
	sort.Slice(d, func(i int, j int) bool {
		return CompareDiagnostic(d[i], d[j]) < 0
	})
}

func CompareDiagnostic(a, b *Diagnostic) int {
	if r := protocol.CompareRange(a.Range, b.Range); r != 0 {
		return r
	}
	if a.Source < b.Source {
		return -1
	}
	if a.Source > b.Source {
		return +1
	}
	if a.Message < b.Message {
		return -1
	}
	if a.Message > b.Message {
		return +1
	}
	return 0
}

// findFileInDeps finds uri in pkg or its dependencies.
func findFileInDeps(pkg Package, uri span.URI) (*ParsedGoFile, Package, error) {
	queue := []Package{pkg}
	seen := make(map[PackageID]bool)

	for len(queue) > 0 {
		pkg := queue[0]
		queue = queue[1:]
		seen[pkg.ID()] = true

		if pgf, err := pkg.File(uri); err == nil {
			return pgf, pkg, nil
		}
		for _, dep := range pkg.Imports() {
			if !seen[dep.ID()] {
				queue = append(queue, dep)
			}
		}
	}
	return nil, nil, fmt.Errorf("no file for %s in package %s", uri, pkg.ID())
}

// UnquoteImportPath returns the unquoted import path of s,
// or "" if the path is not properly quoted.
func UnquoteImportPath(s *ast.ImportSpec) ImportPath {
	path, err := strconv.Unquote(s.Path.Value)
	if err != nil {
		return ""
	}
	return ImportPath(path)
}

// NodeContains returns true if a node encloses a given position pos.
func NodeContains(n ast.Node, pos token.Pos) bool {
	return n != nil && n.Pos() <= pos && pos <= n.End()
}

// CollectScopes returns all scopes in an ast path, ordered as innermost scope
// first.
func CollectScopes(info *types.Info, path []ast.Node, pos token.Pos) []*types.Scope {
	// scopes[i], where i<len(path), is the possibly nil Scope of path[i].
	var scopes []*types.Scope
	for _, n := range path {
		// Include *FuncType scope if pos is inside the function body.
		switch node := n.(type) {
		case *ast.FuncDecl:
			if node.Body != nil && NodeContains(node.Body, pos) {
				n = node.Type
			}
		case *ast.FuncLit:
			if node.Body != nil && NodeContains(node.Body, pos) {
				n = node.Type
			}
		}
		scopes = append(scopes, info.Scopes[n])
	}
	return scopes
}

// Qualifier returns a function that appropriately formats a types.PkgName
// appearing in a *ast.File.
func Qualifier(f *ast.File, pkg *types.Package, info *types.Info) types.Qualifier {
	// Construct mapping of import paths to their defined or implicit names.
	imports := make(map[*types.Package]string)
	for _, imp := range f.Imports {
		var obj types.Object
		if imp.Name != nil {
			obj = info.Defs[imp.Name]
		} else {
			obj = info.Implicits[imp]
		}
		if pkgname, ok := obj.(*types.PkgName); ok {
			imports[pkgname.Imported()] = pkgname.Name()
		}
	}
	// Define qualifier to replace full package paths with names of the imports.
	return func(p *types.Package) string {
		if p == pkg {
			return ""
		}
		if name, ok := imports[p]; ok {
			if name == "." {
				return ""
			}
			return name
		}
		return p.Name()
	}
}

// isDirective reports whether c is a comment directive.
//
// Copied and adapted from go/src/go/ast/ast.go.
func isDirective(c string) bool {
	if len(c) < 3 {
		return false
	}
	if c[1] != '/' {
		return false
	}
	//-style comment (no newline at the end)
	c = c[2:]
	if len(c) == 0 {
		// empty line
		return false
	}
	// "//line " is a line directive.
	// (The // has been removed.)
	if strings.HasPrefix(c, "line ") {
		return true
	}

	// "//[a-z0-9]+:[a-z0-9]"
	// (The // has been removed.)
	colon := strings.Index(c, ":")
	if colon <= 0 || colon+1 >= len(c) {
		return false
	}
	for i := 0; i <= colon+1; i++ {
		if i == colon {
			continue
		}
		b := c[i]
		if !('a' <= b && b <= 'z' || '0' <= b && b <= '9') {
			return false
		}
	}
	return true
}

// InDir checks whether path is in the file tree rooted at dir.
// It checks only the lexical form of the file names.
// It does not consider symbolic links.
//
// Copied from go/src/cmd/go/internal/search/search.go.
func InDir(dir, path string) bool {
	pv := strings.ToUpper(filepath.VolumeName(path))
	dv := strings.ToUpper(filepath.VolumeName(dir))
	path = path[len(pv):]
	dir = dir[len(dv):]
	switch {
	default:
		return false
	case pv != dv:
		return false
	case len(path) == len(dir):
		if path == dir {
			return true
		}
		return false
	case dir == "":
		return path != ""
	case len(path) > len(dir):
		if dir[len(dir)-1] == filepath.Separator {
			if path[:len(dir)] == dir {
				return path[len(dir):] != ""
			}
			return false
		}
		if path[len(dir)] == filepath.Separator && path[:len(dir)] == dir {
			if len(path) == len(dir)+1 {
				return true
			}
			return path[len(dir)+1:] != ""
		}
		return false
	}
}

// IsValidImport returns whether importPkgPath is importable
// by pkgPath
func IsValidImport(pkgPath, importPkgPath PackagePath) bool {
	i := strings.LastIndex(string(importPkgPath), "/internal/")
	if i == -1 {
		return true
	}
	// TODO(rfindley): this looks wrong: IsCommandLineArguments is meant to
	// operate on package IDs, not package paths.
	if IsCommandLineArguments(PackageID(pkgPath)) {
		return true
	}
	// TODO(rfindley): this is wrong. mod.testx/p should not be able to
	// import mod.test/internal: https://go.dev/play/p/-Ca6P-E4V4q
	return strings.HasPrefix(string(pkgPath), string(importPkgPath[:i]))
}

// IsCommandLineArguments reports whether a given value denotes
// "command-line-arguments" package, which is a package with an unknown ID
// created by the go command. It can have a test variant, which is why callers
// should not check that a value equals "command-line-arguments" directly.
func IsCommandLineArguments(id PackageID) bool {
	return strings.Contains(string(id), "command-line-arguments")
}

// RecvIdent returns the type identifier of a method receiver.
// e.g. A for all of A, *A, A[T], *A[T], etc.
func RecvIdent(recv *ast.FieldList) *ast.Ident {
	if recv == nil || len(recv.List) == 0 {
		return nil
	}
	x := recv.List[0].Type
	if star, ok := x.(*ast.StarExpr); ok {
		x = star.X
	}
	switch ix := x.(type) { // check for instantiated receivers
	case *ast.IndexExpr:
		x = ix.X
	case *typeparams.IndexListExpr:
		x = ix.X
	}
	if ident, ok := x.(*ast.Ident); ok {
		return ident
	}
	return nil
}

// embeddedIdent returns the type name identifier for an embedding x, if x in a
// valid embedding. Otherwise, it returns nil.
//
// Spec: An embedded field must be specified as a type name T or as a pointer
// to a non-interface type name *T
func embeddedIdent(x ast.Expr) *ast.Ident {
	if star, ok := x.(*ast.StarExpr); ok {
		x = star.X
	}
	switch ix := x.(type) { // check for instantiated receivers
	case *ast.IndexExpr:
		x = ix.X
	case *typeparams.IndexListExpr:
		x = ix.X
	}
	switch x := x.(type) {
	case *ast.Ident:
		return x
	case *ast.SelectorExpr:
		if _, ok := x.X.(*ast.Ident); ok {
			return x.Sel
		}
	}
	return nil
}
