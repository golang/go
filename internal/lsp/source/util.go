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
	"strings"

	"golang.org/x/tools/internal/lsp/protocol"
	"golang.org/x/tools/internal/span"
	errors "golang.org/x/xerrors"
)

type mappedRange struct {
	spanRange span.Range
	m         *protocol.ColumnMapper

	// protocolRange is the result of converting the spanRange using the mapper.
	// It is computed on-demand.
	protocolRange *protocol.Range
}

func newMappedRange(fset *token.FileSet, m *protocol.ColumnMapper, start, end token.Pos) mappedRange {
	return mappedRange{
		spanRange: span.Range{
			FileSet:   fset,
			Start:     start,
			End:       end,
			Converter: m.Converter,
		},
		m: m,
	}
}

func (s mappedRange) Range() (protocol.Range, error) {
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

func (s mappedRange) Span() (span.Span, error) {
	return s.spanRange.Span()
}

func (s mappedRange) URI() span.URI {
	return s.m.URI
}

// getParsedFile is a convenience function that extracts the Package and ParseGoHandle for a File in a Snapshot.
// selectPackage is typically Narrowest/WidestPackageHandle below.
func getParsedFile(ctx context.Context, snapshot Snapshot, fh FileHandle, selectPackage PackagePolicy) (Package, ParseGoHandle, error) {
	phs, err := snapshot.PackageHandles(ctx, fh)
	if err != nil {
		return nil, nil, err
	}
	ph, err := selectPackage(phs)
	if err != nil {
		return nil, nil, err
	}
	pkg, err := ph.Check(ctx)
	if err != nil {
		return nil, nil, err
	}
	pgh, err := pkg.File(fh.URI())
	return pkg, pgh, err
}

type PackagePolicy func([]PackageHandle) (PackageHandle, error)

// NarrowestPackageHandle picks the "narrowest" package for a given file.
//
// By "narrowest" package, we mean the package with the fewest number of files
// that includes the given file. This solves the problem of test variants,
// as the test will have more files than the non-test package.
func NarrowestPackageHandle(handles []PackageHandle) (PackageHandle, error) {
	if len(handles) < 1 {
		return nil, errors.Errorf("no PackageHandles")
	}
	result := handles[0]
	for _, handle := range handles[1:] {
		if result == nil || len(handle.CompiledGoFiles()) < len(result.CompiledGoFiles()) {
			result = handle
		}
	}
	if result == nil {
		return nil, errors.Errorf("nil PackageHandles have been returned")
	}
	return result, nil
}

// WidestPackageHandle returns the PackageHandle containing the most files.
//
// This is useful for something like diagnostics, where we'd prefer to offer diagnostics
// for as many files as possible.
func WidestPackageHandle(handles []PackageHandle) (PackageHandle, error) {
	if len(handles) < 1 {
		return nil, errors.Errorf("no PackageHandles")
	}
	result := handles[0]
	for _, handle := range handles[1:] {
		if result == nil || len(handle.CompiledGoFiles()) > len(result.CompiledGoFiles()) {
			result = handle
		}
	}
	if result == nil {
		return nil, errors.Errorf("nil PackageHandles have been returned")
	}
	return result, nil
}

// SpecificPackageHandle creates a PackagePolicy to select a
// particular PackageHandle when you alread know the one you want.
func SpecificPackageHandle(desiredID string) PackagePolicy {
	return func(handles []PackageHandle) (PackageHandle, error) {
		for _, h := range handles {
			if h.ID() == desiredID {
				return h, nil
			}
		}

		return nil, fmt.Errorf("no package handle with expected id %q", desiredID)
	}
}

func IsGenerated(ctx context.Context, snapshot Snapshot, uri span.URI) bool {
	fh, err := snapshot.GetFile(ctx, uri)
	if err != nil {
		return false
	}
	ph := snapshot.View().Session().Cache().ParseGoHandle(ctx, fh, ParseHeader)
	parsed, _, _, _, err := ph.Parse(ctx)
	if err != nil {
		return false
	}
	tok := snapshot.View().Session().Cache().FileSet().File(parsed.Pos())
	if tok == nil {
		return false
	}
	for _, commentGroup := range parsed.Comments {
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

func nodeToProtocolRange(view View, pkg Package, n ast.Node) (protocol.Range, error) {
	mrng, err := posToMappedRange(view, pkg, n.Pos(), n.End())
	if err != nil {
		return protocol.Range{}, err
	}
	return mrng.Range()
}

func objToMappedRange(v View, pkg Package, obj types.Object) (mappedRange, error) {
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
			return posToMappedRange(v, pkg, obj.Pos(), obj.Pos()+token.Pos(len(pkgName.Imported().Path())+2))
		}
	}
	return nameToMappedRange(v, pkg, obj.Pos(), obj.Name())
}

func nameToMappedRange(v View, pkg Package, pos token.Pos, name string) (mappedRange, error) {
	return posToMappedRange(v, pkg, pos, pos+token.Pos(len(name)))
}

func posToMappedRange(v View, pkg Package, pos, end token.Pos) (mappedRange, error) {
	logicalFilename := v.Session().Cache().FileSet().File(pos).Position(pos).Filename
	m, err := findMapperInPackage(v, pkg, span.URIFromPath(logicalFilename))
	if err != nil {
		return mappedRange{}, err
	}
	if !pos.IsValid() {
		return mappedRange{}, errors.Errorf("invalid position for %v", pos)
	}
	if !end.IsValid() {
		return mappedRange{}, errors.Errorf("invalid position for %v", end)
	}
	return newMappedRange(v.Session().Cache().FileSet(), m, pos, end), nil
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

// Returns the index and the node whose position is contained inside the node list.
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

// indexExprAtPos returns the index of the expression containing pos.
func exprAtPos(pos token.Pos, args []ast.Expr) int {
	for i, expr := range args {
		if expr.Pos() <= pos && pos <= expr.End() {
			return i
		}
	}
	return len(args)
}

// eachField invokes fn for each field that can be selected from a
// value of type T.
func eachField(T types.Type, fn func(*types.Var)) {
	// TODO(adonovan): this algorithm doesn't exclude ambiguous
	// selections that match more than one field/method.
	// types.NewSelectionSet should do that for us.

	// for termination on recursive types
	var seen map[*types.Struct]bool

	var visit func(T types.Type)
	visit = func(T types.Type) {
		if T, ok := deref(T).Underlying().(*types.Struct); ok {
			if seen[T] {
				return
			}

			for i := 0; i < T.NumFields(); i++ {
				f := T.Field(i)
				fn(f)
				if f.Anonymous() {
					if seen == nil {
						// Lazily create "seen" since it is only needed for
						// embedded structs.
						seen = make(map[*types.Struct]bool)
					}
					seen[T] = true
					visit(f.Type())
				}
			}
		}
	}
	visit(T)
}

// typeIsValid reports whether typ doesn't contain any Invalid types.
func typeIsValid(typ types.Type) bool {
	// Check named types separately, because we don't want
	// to call Underlying() on them to avoid problems with recursive types.
	if _, ok := typ.(*types.Named); ok {
		return true
	}

	switch typ := typ.Underlying().(type) {
	case *types.Basic:
		return typ.Kind() != types.Invalid
	case *types.Array:
		return typeIsValid(typ.Elem())
	case *types.Slice:
		return typeIsValid(typ.Elem())
	case *types.Pointer:
		return typeIsValid(typ.Elem())
	case *types.Map:
		return typeIsValid(typ.Key()) && typeIsValid(typ.Elem())
	case *types.Chan:
		return typeIsValid(typ.Elem())
	case *types.Signature:
		return typeIsValid(typ.Params()) && typeIsValid(typ.Results())
	case *types.Tuple:
		for i := 0; i < typ.Len(); i++ {
			if !typeIsValid(typ.At(i).Type()) {
				return false
			}
		}
		return true
	case *types.Struct, *types.Interface:
		// Don't bother checking structs, interfaces for validity.
		return true
	default:
		return false
	}
}

// resolveInvalid traverses the node of the AST that defines the scope
// containing the declaration of obj, and attempts to find a user-friendly
// name for its invalid type. The resulting Object and its Type are fake.
func resolveInvalid(fset *token.FileSet, obj types.Object, node ast.Node, info *types.Info) types.Object {
	var resultExpr ast.Expr
	ast.Inspect(node, func(node ast.Node) bool {
		switch n := node.(type) {
		case *ast.ValueSpec:
			for _, name := range n.Names {
				if info.Defs[name] == obj {
					resultExpr = n.Type
				}
			}
			return false
		case *ast.Field: // This case handles parameters and results of a FuncDecl or FuncLit.
			for _, name := range n.Names {
				if info.Defs[name] == obj {
					resultExpr = n.Type
				}
			}
			return false
		default:
			return true
		}
	})
	// Construct a fake type for the object and return a fake object with this type.
	typename := formatNode(fset, resultExpr)
	typ := types.NewNamed(types.NewTypeName(token.NoPos, obj.Pkg(), typename, nil), types.Typ[types.Invalid], nil)
	return types.NewVar(obj.Pos(), obj.Pkg(), obj.Name(), typ)
}

func formatNode(fset *token.FileSet, n ast.Node) string {
	var buf strings.Builder
	if err := printer.Fprint(&buf, fset, n); err != nil {
		return ""
	}
	return buf.String()
}

func isPointer(T types.Type) bool {
	_, ok := T.(*types.Pointer)
	return ok
}

func isVar(obj types.Object) bool {
	_, ok := obj.(*types.Var)
	return ok
}

// deref returns a pointer's element type, traversing as many levels as needed.
// Otherwise it returns typ.
func deref(typ types.Type) types.Type {
	for {
		p, ok := typ.Underlying().(*types.Pointer)
		if !ok {
			return typ
		}
		typ = p.Elem()
	}
}

func isTypeName(obj types.Object) bool {
	_, ok := obj.(*types.TypeName)
	return ok
}

func isFunc(obj types.Object) bool {
	_, ok := obj.(*types.Func)
	return ok
}

func isEmptyInterface(T types.Type) bool {
	intf, _ := T.(*types.Interface)
	return intf != nil && intf.NumMethods() == 0
}

func isUntyped(T types.Type) bool {
	if basic, ok := T.(*types.Basic); ok {
		return basic.Info()&types.IsUntyped > 0
	}
	return false
}

func isPkgName(obj types.Object) bool {
	_, ok := obj.(*types.PkgName)
	return ok
}

func isASTFile(n ast.Node) bool {
	_, ok := n.(*ast.File)
	return ok
}

func deslice(T types.Type) types.Type {
	if slice, ok := T.Underlying().(*types.Slice); ok {
		return slice.Elem()
	}
	return nil
}

// isSelector returns the enclosing *ast.SelectorExpr when pos is in the
// selector.
func enclosingSelector(path []ast.Node, pos token.Pos) *ast.SelectorExpr {
	if len(path) == 0 {
		return nil
	}

	if sel, ok := path[0].(*ast.SelectorExpr); ok {
		return sel
	}

	if _, ok := path[0].(*ast.Ident); ok && len(path) > 1 {
		if sel, ok := path[1].(*ast.SelectorExpr); ok && pos >= sel.Sel.Pos() {
			return sel
		}
	}

	return nil
}

func enclosingValueSpec(path []ast.Node) *ast.ValueSpec {
	for _, n := range path {
		if vs, ok := n.(*ast.ValueSpec); ok {
			return vs
		}
	}

	return nil
}

// typeConversion returns the type being converted to if call is a type
// conversion expression.
func typeConversion(call *ast.CallExpr, info *types.Info) types.Type {
	var ident *ast.Ident
	switch expr := call.Fun.(type) {
	case *ast.Ident:
		ident = expr
	case *ast.SelectorExpr:
		ident = expr.Sel
	default:
		return nil
	}

	// Type conversion (e.g. "float64(foo)").
	if fun, _ := info.ObjectOf(ident).(*types.TypeName); fun != nil {
		return fun.Type()
	}

	return nil
}

// fieldsAccessible returns whether s has at least one field accessible by p.
func fieldsAccessible(s *types.Struct, p *types.Package) bool {
	for i := 0; i < s.NumFields(); i++ {
		f := s.Field(i)
		if f.Exported() || f.Pkg() == p {
			return true
		}
	}
	return false
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

func findPosInPackage(v View, searchpkg Package, pos token.Pos) (*ast.File, Package, error) {
	tok := v.Session().Cache().FileSet().File(pos)
	if tok == nil {
		return nil, nil, errors.Errorf("no file for pos in package %s", searchpkg.ID())
	}
	uri := span.URIFromPath(tok.Name())

	ph, pkg, err := FindFileInPackage(searchpkg, uri)
	if err != nil {
		return nil, nil, err
	}
	file, _, _, _, err := ph.Cached()
	if err != nil {
		return nil, nil, err
	}
	if !(file.Pos() <= pos && pos <= file.End()) {
		return nil, nil, fmt.Errorf("pos %v, apparently in file %q, is not between %v and %v", pos, ph.File().URI(), file.Pos(), file.End())
	}
	return file, pkg, nil
}

func findMapperInPackage(v View, searchpkg Package, uri span.URI) (*protocol.ColumnMapper, error) {
	ph, _, err := FindFileInPackage(searchpkg, uri)
	if err != nil {
		return nil, err
	}
	_, _, m, _, err := ph.Cached()
	if err != nil {
		return nil, err
	}
	return m, nil
}

// FindFileInPackage finds uri in pkg or its dependencies.
func FindFileInPackage(pkg Package, uri span.URI) (ParseGoHandle, Package, error) {
	queue := []Package{pkg}
	seen := make(map[string]bool)

	for len(queue) > 0 {
		pkg := queue[0]
		queue = queue[1:]
		seen[pkg.ID()] = true

		if f, err := pkg.File(uri); err == nil {
			return f, pkg, nil
		}
		for _, dep := range pkg.Imports() {
			if !seen[dep.ID()] {
				queue = append(queue, dep)
			}
		}
	}
	return nil, nil, errors.Errorf("no file for %s in package %s", uri, pkg.ID())
}

// prevStmt returns the statement that precedes the statement containing pos.
// For example:
//
//     foo := 1
//     bar(1 + 2<>)
//
// If "<>" is pos, prevStmt returns "foo := 1"
func prevStmt(pos token.Pos, path []ast.Node) ast.Stmt {
	var blockLines []ast.Stmt
	for i := 0; i < len(path) && blockLines == nil; i++ {
		switch n := path[i].(type) {
		case *ast.BlockStmt:
			blockLines = n.List
		case *ast.CommClause:
			blockLines = n.Body
		case *ast.CaseClause:
			blockLines = n.Body
		}
	}

	for i := len(blockLines) - 1; i >= 0; i-- {
		if blockLines[i].End() < pos {
			return blockLines[i]
		}
	}

	return nil
}

// formatZeroValue produces Go code representing the zero value of T.
func formatZeroValue(T types.Type, qf types.Qualifier) string {
	switch u := T.Underlying().(type) {
	case *types.Basic:
		switch {
		case u.Info()&types.IsNumeric > 0:
			return "0"
		case u.Info()&types.IsString > 0:
			return `""`
		case u.Info()&types.IsBoolean > 0:
			return "false"
		default:
			panic(fmt.Sprintf("unhandled basic type: %v", u))
		}
	case *types.Pointer, *types.Interface, *types.Chan, *types.Map, *types.Slice, *types.Signature:
		return "nil"
	default:
		return types.TypeString(T, qf) + "{}"
	}
}
