// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"go/ast"
	"go/printer"
	"go/token"
	"go/types"
	"path/filepath"
	"regexp"
	"sort"
	"strconv"
	"strings"

	"golang.org/x/tools/gopls/internal/bug"
	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/gopls/internal/lsp/safetoken"
	"golang.org/x/tools/gopls/internal/span"
	"golang.org/x/tools/internal/tokeninternal"
	"golang.org/x/tools/internal/typeparams"
)

// IsGenerated gets and reads the file denoted by uri and reports
// whether it contains a "generated file" comment as described at
// https://golang.org/s/generatedcode.
//
// TODO(adonovan): opt: this function does too much.
// Move snapshot.ReadFile into the caller (most of which have already done it).
func IsGenerated(ctx context.Context, snapshot Snapshot, uri span.URI) bool {
	fh, err := snapshot.ReadFile(ctx, uri)
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
				if safetoken.Position(pgf.Tok, comment.Slash).Column == 1 {
					return true
				}
			}
		}
	}
	return false
}

// adjustedObjEnd returns the end position of obj, possibly modified for
// package names.
//
// TODO(rfindley): eliminate this function, by inlining it at callsites where
// it makes sense.
func adjustedObjEnd(obj types.Object) token.Pos {
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
	return obj.Pos() + token.Pos(nameLen)
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

// FormatNode returns the "pretty-print" output for an ast node.
func FormatNode(fset *token.FileSet, n ast.Node) string {
	var buf strings.Builder
	if err := printer.Fprint(&buf, fset, n); err != nil {
		// TODO(rfindley): we should use bug.Reportf here.
		// We encounter this during completion.resolveInvalid.
		return ""
	}
	return buf.String()
}

// FormatNodeFile is like FormatNode, but requires only the token.File for the
// syntax containing the given ast node.
func FormatNodeFile(file *token.File, n ast.Node) string {
	fset := tokeninternal.FileSetFor(file)
	return FormatNode(fset, n)
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

// findFileInDeps finds package metadata containing URI in the transitive
// dependencies of m. When using the Go command, the answer is unique.
//
// TODO(rfindley): refactor to share logic with findPackageInDeps?
func findFileInDeps(s MetadataSource, m *Metadata, uri span.URI) *Metadata {
	seen := make(map[PackageID]bool)
	var search func(*Metadata) *Metadata
	search = func(m *Metadata) *Metadata {
		if seen[m.ID] {
			return nil
		}
		seen[m.ID] = true
		for _, cgf := range m.CompiledGoFiles {
			if cgf == uri {
				return m
			}
		}
		for _, dep := range m.DepsByPkgPath {
			m := s.Metadata(dep)
			if m == nil {
				bug.Reportf("nil metadata for %q", dep)
				continue
			}
			if found := search(m); found != nil {
				return found
			}
		}
		return nil
	}
	return search(m)
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
		if pkgname, ok := ImportedPkgName(info, imp); ok {
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

// requalifier returns a function that re-qualifies identifiers and qualified
// identifiers contained in targetFile using the given metadata qualifier.
func requalifier(s MetadataSource, targetFile *ast.File, targetMeta *Metadata, mq MetadataQualifier) func(string) string {
	qm := map[string]string{
		"": mq(targetMeta.Name, "", targetMeta.PkgPath),
	}

	// Construct mapping of import paths to their defined or implicit names.
	for _, imp := range targetFile.Imports {
		name, pkgName, impPath, pkgPath := importInfo(s, imp, targetMeta)

		// Re-map the target name for the source file.
		qm[name] = mq(pkgName, impPath, pkgPath)
	}

	return func(name string) string {
		if newName, ok := qm[name]; ok {
			return newName
		}
		return name
	}
}

// A MetadataQualifier is a function that qualifies an identifier declared in a
// package with the given package name, import path, and package path.
//
// In scenarios where metadata is missing the provided PackageName and
// PackagePath may be empty, but ImportPath must always be non-empty.
type MetadataQualifier func(PackageName, ImportPath, PackagePath) string

// MetadataQualifierForFile returns a metadata qualifier that chooses the best
// qualification of an imported package relative to the file f in package with
// metadata m.
func MetadataQualifierForFile(s MetadataSource, f *ast.File, m *Metadata) MetadataQualifier {
	// Record local names for import paths.
	localNames := make(map[ImportPath]string) // local names for imports in f
	for _, imp := range f.Imports {
		name, _, impPath, _ := importInfo(s, imp, m)
		localNames[impPath] = name
	}

	// Record a package path -> import path mapping.
	inverseDeps := make(map[PackageID]PackagePath)
	for path, id := range m.DepsByPkgPath {
		inverseDeps[id] = path
	}
	importsByPkgPath := make(map[PackagePath]ImportPath) // best import paths by pkgPath
	for impPath, id := range m.DepsByImpPath {
		if id == "" {
			continue
		}
		pkgPath := inverseDeps[id]
		_, hasPath := importsByPkgPath[pkgPath]
		_, hasImp := localNames[impPath]
		// In rare cases, there may be multiple import paths with the same package
		// path. In such scenarios, prefer an import path that already exists in
		// the file.
		if !hasPath || hasImp {
			importsByPkgPath[pkgPath] = impPath
		}
	}

	return func(pkgName PackageName, impPath ImportPath, pkgPath PackagePath) string {
		// If supplied, translate the package path to an import path in the source
		// package.
		if pkgPath != "" {
			if srcImp := importsByPkgPath[pkgPath]; srcImp != "" {
				impPath = srcImp
			}
			if pkgPath == m.PkgPath {
				return ""
			}
		}
		if localName, ok := localNames[impPath]; ok && impPath != "" {
			return string(localName)
		}
		if pkgName != "" {
			return string(pkgName)
		}
		idx := strings.LastIndexByte(string(impPath), '/')
		return string(impPath[idx+1:])
	}
}

// importInfo collects information about the import specified by imp,
// extracting its file-local name, package name, import path, and package path.
//
// If metadata is missing for the import, the resulting package name and
// package path may be empty, and the file local name may be guessed based on
// the import path.
//
// Note: previous versions of this helper used a PackageID->PackagePath map
// extracted from m, for extracting package path even in the case where
// metadata for a dep was missing. This should not be necessary, as we should
// always have metadata for IDs contained in DepsByPkgPath.
func importInfo(s MetadataSource, imp *ast.ImportSpec, m *Metadata) (string, PackageName, ImportPath, PackagePath) {
	var (
		name    string // local name
		pkgName PackageName
		impPath = UnquoteImportPath(imp)
		pkgPath PackagePath
	)

	// If the import has a local name, use it.
	if imp.Name != nil {
		name = imp.Name.Name
	}

	// Try to find metadata for the import. If successful and there is no local
	// name, the package name is the local name.
	if depID := m.DepsByImpPath[impPath]; depID != "" {
		if depm := s.Metadata(depID); depm != nil {
			if name == "" {
				name = string(depm.Name)
			}
			pkgName = depm.Name
			pkgPath = depm.PkgPath
		}
	}

	// If the local name is still unknown, guess it based on the import path.
	if name == "" {
		idx := strings.LastIndexByte(string(impPath), '/')
		name = string(impPath[idx+1:])
	}
	return name, pkgName, impPath, pkgPath
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

// An importFunc is an implementation of the single-method
// types.Importer interface based on a function value.
type ImporterFunc func(path string) (*types.Package, error)

func (f ImporterFunc) Import(path string) (*types.Package, error) { return f(path) }
