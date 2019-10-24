// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"context"
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
	"path/filepath"
	"regexp"
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

// NarrowestCheckPackageHandle picks the "narrowest" package for a given file.
//
// By "narrowest" package, we mean the package with the fewest number of files
// that includes the given file. This solves the problem of test variants,
// as the test will have more files than the non-test package.
func NarrowestCheckPackageHandle(handles []CheckPackageHandle) (CheckPackageHandle, error) {
	if len(handles) < 1 {
		return nil, errors.Errorf("no CheckPackageHandles")
	}
	result := handles[0]
	for _, handle := range handles[1:] {
		if result == nil || len(handle.Files()) < len(result.Files()) {
			result = handle
		}
	}
	if result == nil {
		return nil, errors.Errorf("nil CheckPackageHandles have been returned")
	}
	return result, nil
}

// WidestCheckPackageHandle returns the CheckPackageHandle containing the most files.
//
// This is useful for something like diagnostics, where we'd prefer to offer diagnostics
// for as many files as possible.
func WidestCheckPackageHandle(handles []CheckPackageHandle) (CheckPackageHandle, error) {
	if len(handles) < 1 {
		return nil, errors.Errorf("no CheckPackageHandles")
	}
	result := handles[0]
	for _, handle := range handles[1:] {
		if result == nil || len(handle.Files()) > len(result.Files()) {
			result = handle
		}
	}
	if result == nil {
		return nil, errors.Errorf("nil CheckPackageHandles have been returned")
	}
	return result, nil
}

func IsGenerated(ctx context.Context, view View, uri span.URI) bool {
	f, err := view.GetFile(ctx, uri)
	if err != nil {
		return false
	}
	ph := view.Session().Cache().ParseGoHandle(view.Snapshot().Handle(ctx, f), ParseHeader)
	parsed, _, _, err := ph.Parse(ctx)
	if err != nil {
		return false
	}
	tok := view.Session().Cache().FileSet().File(parsed.Pos())
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

func nodeToProtocolRange(ctx context.Context, view View, m *protocol.ColumnMapper, n ast.Node) (protocol.Range, error) {
	mrng, err := nodeToMappedRange(ctx, view, m, n)
	if err != nil {
		return protocol.Range{}, err
	}
	return mrng.Range()
}

func objToMappedRange(ctx context.Context, pkg Package, obj types.Object) (mappedRange, error) {
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
		// of the object be the point at the beginning of the declaration.
		if pkgName.Imported().Name() == pkgName.Name() {
			return nameToMappedRange(ctx, pkg, obj.Pos(), "")
		}
	}
	return nameToMappedRange(ctx, pkg, obj.Pos(), obj.Name())
}

func nameToMappedRange(ctx context.Context, pkg Package, pos token.Pos, name string) (mappedRange, error) {
	return posToMappedRange(ctx, pkg, pos, pos+token.Pos(len(name)))
}

func nodeToMappedRange(ctx context.Context, view View, m *protocol.ColumnMapper, n ast.Node) (mappedRange, error) {
	return posToRange(ctx, view, m, n.Pos(), n.End())
}

func posToMappedRange(ctx context.Context, pkg Package, pos, end token.Pos) (mappedRange, error) {
	m, err := posToMapper(ctx, pkg, pos)
	if err != nil {
		return mappedRange{}, err
	}
	return posToRange(ctx, pkg.View(), m, pos, end)
}

func posToRange(ctx context.Context, view View, m *protocol.ColumnMapper, pos, end token.Pos) (mappedRange, error) {
	if !pos.IsValid() {
		return mappedRange{}, errors.Errorf("invalid position for %v", pos)
	}
	if !end.IsValid() {
		return mappedRange{}, errors.Errorf("invalid position for %v", end)
	}
	return mappedRange{
		m:         m,
		spanRange: span.NewRange(view.Session().Cache().FileSet(), pos, end),
	}, nil
}

func posToMapper(ctx context.Context, pkg Package, pos token.Pos) (*protocol.ColumnMapper, error) {
	posn := pkg.View().Session().Cache().FileSet().Position(pos)
	ph, _, err := pkg.FindFile(ctx, span.FileURI(posn.Filename))
	if err != nil {
		return nil, err
	}
	_, m, _, err := ph.Cached()
	return m, err
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

// indexExprAtPos returns the index of the expression containing pos.
func indexExprAtPos(pos token.Pos, args []ast.Expr) int {
	for i, expr := range args {
		if expr.Pos() <= pos && pos <= expr.End() {
			return i
		}
	}
	return len(args)
}

func exprAtPos(pos token.Pos, args []ast.Expr) ast.Expr {
	for _, expr := range args {
		if expr.Pos() <= pos && pos <= expr.End() {
			return expr
		}
	}
	return nil
}

// fieldSelections returns the set of fields that can
// be selected from a value of type T.
func fieldSelections(T types.Type) (fields []*types.Var) {
	// TODO(adonovan): this algorithm doesn't exclude ambiguous
	// selections that match more than one field/method.
	// types.NewSelectionSet should do that for us.

	seen := make(map[*types.Var]bool) // for termination on recursive types

	var visit func(T types.Type)
	visit = func(T types.Type) {
		if T, ok := deref(T).Underlying().(*types.Struct); ok {
			for i := 0; i < T.NumFields(); i++ {
				f := T.Field(i)
				if seen[f] {
					continue
				}
				seen[f] = true
				fields = append(fields, f)
				if f.Anonymous() {
					visit(f.Type())
				}
			}
		}
	}
	visit(T)

	return fields
}

// resolveInvalid traverses the node of the AST that defines the scope
// containing the declaration of obj, and attempts to find a user-friendly
// name for its invalid type. The resulting Object and its Type are fake.
func resolveInvalid(obj types.Object, node ast.Node, info *types.Info) types.Object {
	// Construct a fake type for the object and return a fake object with this type.
	formatResult := func(expr ast.Expr) types.Object {
		var typename string
		switch t := expr.(type) {
		case *ast.SelectorExpr:
			typename = fmt.Sprintf("%s.%s", t.X, t.Sel)
		case *ast.Ident:
			typename = t.String()
		default:
			return nil
		}
		typ := types.NewNamed(types.NewTypeName(token.NoPos, obj.Pkg(), typename, nil), types.Typ[types.Invalid], nil)
		return types.NewVar(obj.Pos(), obj.Pkg(), obj.Name(), typ)
	}
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
		// TODO(rstambler): Handle range statements.
		default:
			return true
		}
	})
	return formatResult(resultExpr)
}

func isPointer(T types.Type) bool {
	_, ok := T.(*types.Pointer)
	return ok
}

// deref returns a pointer's element type; otherwise it returns typ.
func deref(typ types.Type) types.Type {
	if p, ok := typ.Underlying().(*types.Pointer); ok {
		return p.Elem()
	}
	return typ
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

func formatParams(tup *types.Tuple, variadic bool, qf types.Qualifier) []string {
	params := make([]string, 0, tup.Len())
	for i := 0; i < tup.Len(); i++ {
		el := tup.At(i)
		typ := types.TypeString(el.Type(), qf)

		// Handle a variadic parameter (can only be the final parameter).
		if variadic && i == tup.Len()-1 {
			typ = strings.Replace(typ, "[]", "...", 1)
		}

		if el.Name() == "" {
			params = append(params, typ)
		} else {
			params = append(params, el.Name()+" "+typ)
		}
	}
	return params
}

func formatResults(tup *types.Tuple, qf types.Qualifier) ([]string, bool) {
	var writeResultParens bool
	results := make([]string, 0, tup.Len())
	for i := 0; i < tup.Len(); i++ {
		if i >= 1 {
			writeResultParens = true
		}
		el := tup.At(i)
		typ := types.TypeString(el.Type(), qf)

		if el.Name() == "" {
			results = append(results, typ)
		} else {
			if i == 0 {
				writeResultParens = true
			}
			results = append(results, el.Name()+" "+typ)
		}
	}
	return results, writeResultParens
}

// formatType returns the detail and kind for an object of type *types.TypeName.
func formatType(typ types.Type, qf types.Qualifier) (detail string, kind protocol.CompletionItemKind) {
	if types.IsInterface(typ) {
		detail = "interface{...}"
		kind = protocol.InterfaceCompletion
	} else if _, ok := typ.(*types.Struct); ok {
		detail = "struct{...}"
		kind = protocol.StructCompletion
	} else if typ != typ.Underlying() {
		detail, kind = formatType(typ.Underlying(), qf)
	} else {
		detail = types.TypeString(typ, qf)
		kind = protocol.ClassCompletion
	}
	return detail, kind
}

func formatFunction(params []string, results []string, writeResultParens bool) string {
	var detail strings.Builder

	detail.WriteByte('(')
	for i, p := range params {
		if i > 0 {
			detail.WriteString(", ")
		}
		detail.WriteString(p)
	}
	detail.WriteByte(')')

	// Add space between parameters and results.
	if len(results) > 0 {
		detail.WriteByte(' ')
	}

	if writeResultParens {
		detail.WriteByte('(')
	}
	for i, p := range results {
		if i > 0 {
			detail.WriteString(", ")
		}
		detail.WriteString(p)
	}
	if writeResultParens {
		detail.WriteByte(')')
	}

	return detail.String()
}
