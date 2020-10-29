// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"encoding/json"
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

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

// MappedRange provides mapped protocol.Range for a span.Range, accounting for
// UTF-16 code points.
type MappedRange struct {
	spanRange span.Range
	m         *protocol.ColumnMapper

	// protocolRange is the result of converting the spanRange using the mapper.
	// It is computed on-demand.
	protocolRange *protocol.Range
}

// NewMappedRange returns a MappedRange for the given start and end token.Pos.
func NewMappedRange(fset *token.FileSet, m *protocol.ColumnMapper, start, end token.Pos) MappedRange {
	return MappedRange{
		spanRange: span.Range{
			FileSet:   fset,
			Start:     start,
			End:       end,
			Converter: m.Converter,
		},
		m: m,
	}
}

func (s MappedRange) Range() (protocol.Range, error) {
	if s.protocolRange == nil {
		spn, err := s.spanRange.Span()
		if err != nil {
			return protocol.Range{}, err
		}
		prng, err := s.m.Range(spn)
		if err != nil {
			return protocol.Range{}, err
		}
		s.protocolRange = &prng
	}
	return *s.protocolRange, nil
}

func (s MappedRange) Span() (span.Span, error) {
	return s.spanRange.Span()
}

func (s MappedRange) SpanRange() span.Range {
	return s.spanRange
}

func (s MappedRange) URI() span.URI {
	return s.m.URI
}

// GetParsedFile is a convenience function that extracts the Package and
// ParsedGoFile for a file in a Snapshot. pkgPolicy is one of NarrowestPackage/
// WidestPackage.
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
	tok := snapshot.FileSet().File(pgf.File.Pos())
	if tok == nil {
		return false
	}
	for _, commentGroup := range pgf.File.Comments {
		for _, comment := range commentGroup.List {
			if matched := generatedRx.MatchString(comment.Text); matched {
				// Check if comment is at the beginning of the line in source.
				if pos := tok.Position(comment.Slash); pos.Column == 1 {
					return true
				}
			}
		}
	}
	return false
}

func nodeToProtocolRange(snapshot Snapshot, pkg Package, n ast.Node) (protocol.Range, error) {
	mrng, err := posToMappedRange(snapshot, pkg, n.Pos(), n.End())
	if err != nil {
		return protocol.Range{}, err
	}
	return mrng.Range()
}

func objToMappedRange(snapshot Snapshot, pkg Package, obj types.Object) (MappedRange, error) {
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
			return posToMappedRange(snapshot, pkg, obj.Pos(), obj.Pos()+token.Pos(len(pkgName.Imported().Path())+2))
		}
	}
	return nameToMappedRange(snapshot, pkg, obj.Pos(), obj.Name())
}

func nameToMappedRange(snapshot Snapshot, pkg Package, pos token.Pos, name string) (MappedRange, error) {
	return posToMappedRange(snapshot, pkg, pos, pos+token.Pos(len(name)))
}

func posToMappedRange(snapshot Snapshot, pkg Package, pos, end token.Pos) (MappedRange, error) {
	logicalFilename := snapshot.FileSet().File(pos).Position(pos).Filename
	pgf, _, err := findFileInDeps(pkg, span.URIFromPath(logicalFilename))
	if err != nil {
		return MappedRange{}, err
	}
	if !pos.IsValid() {
		return MappedRange{}, errors.Errorf("invalid position for %v", pos)
	}
	if !end.IsValid() {
		return MappedRange{}, errors.Errorf("invalid position for %v", end)
	}
	return NewMappedRange(snapshot.FileSet(), pgf.Mapper, pos, end), nil
}

// Matches cgo generated comment as well as the proposed standard:
//	https://golang.org/s/generatedcode
var generatedRx = regexp.MustCompile(`// .*DO NOT EDIT\.?`)

func DetectLanguage(langID, filename string) FileKind {
	switch langID {
	case "go":
		return Go
	case "go.mod":
		return Mod
	case "go.sum":
		return Sum
	}
	// Fallback to detecting the language based on the file extension.
	switch filepath.Ext(filename) {
	case ".mod":
		return Mod
	case ".sum":
		return Sum
	default: // fallback to Go
		return Go
	}
}

func (k FileKind) String() string {
	switch k {
	case Mod:
		return "go.mod"
	case Sum:
		return "go.sum"
	default:
		return "go"
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
func Deref(typ types.Type) types.Type {
	for {
		p, ok := typ.Underlying().(*types.Pointer)
		if !ok {
			return typ
		}
		typ = p.Elem()
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
	if a.Message < b.Message {
		return -1
	}
	if a.Message == b.Message {
		return 0
	}
	return 1
}

// FindPosInPackage finds the parsed file for a position in a given search
// package.
func FindPosInPackage(snapshot Snapshot, searchpkg Package, pos token.Pos) (*ParsedGoFile, Package, error) {
	tok := snapshot.FileSet().File(pos)
	if tok == nil {
		return nil, nil, errors.Errorf("no file for pos in package %s", searchpkg.ID())
	}
	uri := span.URIFromPath(tok.Name())

	pgf, pkg, err := findFileInDeps(searchpkg, uri)
	if err != nil {
		return nil, nil, err
	}
	return pgf, pkg, nil
}

// findFileInDeps finds uri in pkg or its dependencies.
func findFileInDeps(pkg Package, uri span.URI) (*ParsedGoFile, Package, error) {
	queue := []Package{pkg}
	seen := make(map[string]bool)

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
	return nil, nil, errors.Errorf("no file for %s in package %s", uri, pkg.ID())
}

// MarshalArgs encodes the given arguments to json.RawMessages. This function
// is used to construct arguments to a protocol.Command.
//
// Example usage:
//
//   jsonArgs, err := EncodeArgs(1, "hello", true, StructuredArg{42, 12.6})
//
func MarshalArgs(args ...interface{}) ([]json.RawMessage, error) {
	var out []json.RawMessage
	for _, arg := range args {
		argJSON, err := json.Marshal(arg)
		if err != nil {
			return nil, err
		}
		out = append(out, argJSON)
	}
	return out, nil
}

// UnmarshalArgs decodes the given json.RawMessages to the variables provided
// by args. Each element of args should be a pointer.
//
// Example usage:
//
//   var (
//       num int
//       str string
//       bul bool
//       structured StructuredArg
//   )
//   err := UnmarshalArgs(args, &num, &str, &bul, &structured)
//
func UnmarshalArgs(jsonArgs []json.RawMessage, args ...interface{}) error {
	if len(args) != len(jsonArgs) {
		return fmt.Errorf("DecodeArgs: expected %d input arguments, got %d JSON arguments", len(args), len(jsonArgs))
	}
	for i, arg := range args {
		if err := json.Unmarshal(jsonArgs[i], arg); err != nil {
			return err
		}
	}
	return nil
}

// ImportPath returns the unquoted import path of s,
// or "" if the path is not properly quoted.
func ImportPath(s *ast.ImportSpec) string {
	t, err := strconv.Unquote(s.Path.Value)
	if err != nil {
		return ""
	}
	return t
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
// InDir makes some effort to succeed even in the presence of symbolic links.
//
// Copied and slightly adjusted from go/src/cmd/go/internal/search/search.go.
func InDir(dir, path string) bool {
	if InDirLex(dir, path) {
		return true
	}
	xpath, err := filepath.EvalSymlinks(path)
	if err != nil || xpath == path {
		xpath = ""
	} else {
		if InDirLex(dir, xpath) {
			return true
		}
	}

	xdir, err := filepath.EvalSymlinks(dir)
	if err == nil && xdir != dir {
		if InDirLex(xdir, path) {
			return true
		}
		if xpath != "" {
			if InDirLex(xdir, xpath) {
				return true
			}
		}
	}
	return false
}

// InDirLex is like inDir but only checks the lexical form of the file names.
// It does not consider symbolic links.
//
// Copied from go/src/cmd/go/internal/search/search.go.
func InDirLex(dir, path string) bool {
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
