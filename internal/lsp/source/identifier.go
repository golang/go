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
	"strconv"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/protocol"
	errors "golang.org/x/xerrors"
)

// IdentifierInfo holds information about an identifier in Go source.
type IdentifierInfo struct {
	Name     string
	Snapshot Snapshot
	mappedRange

	Type struct {
		mappedRange
		Object types.Object
	}

	Declaration Declaration

	ident *ast.Ident

	// enclosing is an expression used to determine the link anchor for an identifier.
	enclosing types.Type

	pkg Package
	qf  types.Qualifier
}

type Declaration struct {
	MappedRange []mappedRange
	node        ast.Node
	obj         types.Object
}

// Identifier returns identifier information for a position
// in a file, accounting for a potentially incomplete selector.
func Identifier(ctx context.Context, snapshot Snapshot, fh FileHandle, pos protocol.Position) (*IdentifierInfo, error) {
	ctx, done := event.Start(ctx, "source.Identifier")
	defer done()

	pkg, pgh, err := getParsedFile(ctx, snapshot, fh, NarrowestPackageHandle)
	if err != nil {
		return nil, fmt.Errorf("getting file for Identifier: %w", err)
	}
	file, _, m, _, err := pgh.Cached()
	if err != nil {
		return nil, err
	}
	spn, err := m.PointSpan(pos)
	if err != nil {
		return nil, err
	}
	rng, err := spn.Range(m.Converter)
	if err != nil {
		return nil, err
	}
	return findIdentifier(ctx, snapshot, pkg, file, rng.Start)
}

var ErrNoIdentFound = errors.New("no identifier found")

func findIdentifier(ctx context.Context, s Snapshot, pkg Package, file *ast.File, pos token.Pos) (*IdentifierInfo, error) {
	// Handle import specs separately, as there is no formal position for a package declaration.
	if result, err := importSpec(s, pkg, file, pos); result != nil || err != nil {
		return result, err
	}
	path := pathEnclosingObjNode(file, pos)
	if path == nil {
		return nil, ErrNoIdentFound
	}

	view := s.View()
	qf := qualifier(file, pkg.GetTypes(), pkg.GetTypesInfo())

	ident, _ := path[0].(*ast.Ident)
	if ident == nil {
		return nil, ErrNoIdentFound
	}
	// Special case for package declarations, since they have no
	// corresponding types.Object.
	if ident == file.Name {
		rng, err := posToMappedRange(view, pkg, file.Name.Pos(), file.Name.End())
		if err != nil {
			return nil, err
		}
		var declAST *ast.File
		for _, f := range pkg.GetSyntax() {
			if f.Doc != nil {
				declAST = f
			}
		}
		// If there's no package documentation, just use current file.
		if declAST == nil {
			declAST = file
		}
		declRng, err := posToMappedRange(view, pkg, declAST.Name.Pos(), declAST.Name.End())
		if err != nil {
			return nil, err
		}
		return &IdentifierInfo{
			Name:        file.Name.Name,
			ident:       file.Name,
			mappedRange: rng,
			pkg:         pkg,
			qf:          qf,
			Snapshot:    s,
			Declaration: Declaration{
				node:        declAST.Name,
				MappedRange: []mappedRange{declRng},
			},
		}, nil
	}

	result := &IdentifierInfo{
		Snapshot:  s,
		qf:        qf,
		pkg:       pkg,
		ident:     ident,
		enclosing: searchForEnclosing(pkg, path),
	}

	var wasEmbeddedField bool
	for _, n := range path[1:] {
		if field, ok := n.(*ast.Field); ok {
			wasEmbeddedField = len(field.Names) == 0
			break
		}
	}

	result.Name = result.ident.Name
	var err error
	if result.mappedRange, err = posToMappedRange(view, pkg, result.ident.Pos(), result.ident.End()); err != nil {
		return nil, err
	}

	result.Declaration.obj = pkg.GetTypesInfo().ObjectOf(result.ident)
	if result.Declaration.obj == nil {
		// If there was no types.Object for the declaration, there might be an implicit local variable
		// declaration in a type switch.
		if objs := typeSwitchImplicits(pkg, path); len(objs) > 0 {
			// There is no types.Object for the declaration of an implicit local variable,
			// but all of the types.Objects associated with the usages of this variable can be
			// used to connect it back to the declaration.
			// Preserve the first of these objects and treat it as if it were the declaring object.
			result.Declaration.obj = objs[0]
		} else {
			// Probably a type error.
			return nil, errors.Errorf("no object for ident %v", result.Name)
		}
	}

	// Handle builtins separately.
	if result.Declaration.obj.Parent() == types.Universe {
		astObj, err := view.LookupBuiltin(ctx, result.Name)
		if err != nil {
			return nil, err
		}
		decl, ok := astObj.Decl.(ast.Node)
		if !ok {
			return nil, errors.Errorf("no declaration for %s", result.Name)
		}
		result.Declaration.node = decl

		rng, err := nameToMappedRange(view, pkg, decl.Pos(), result.Name)
		if err != nil {
			return nil, err
		}
		result.Declaration.MappedRange = append(result.Declaration.MappedRange, rng)

		return result, nil
	}

	if wasEmbeddedField {
		// The original position was on the embedded field declaration, so we
		// try to dig out the type and jump to that instead.
		if v, ok := result.Declaration.obj.(*types.Var); ok {
			if typObj := typeToObject(v.Type()); typObj != nil {
				result.Declaration.obj = typObj
			}
		}
	}

	rng, err := objToMappedRange(view, pkg, result.Declaration.obj)
	if err != nil {
		return nil, err
	}
	result.Declaration.MappedRange = append(result.Declaration.MappedRange, rng)

	if result.Declaration.node, err = objToNode(s.View(), pkg, result.Declaration.obj); err != nil {
		return nil, err
	}
	typ := pkg.GetTypesInfo().TypeOf(result.ident)
	if typ == nil {
		return result, nil
	}

	result.Type.Object = typeToObject(typ)
	if result.Type.Object != nil {
		// Identifiers with the type "error" are a special case with no position.
		if hasErrorType(result.Type.Object) {
			return result, nil
		}
		if result.Type.mappedRange, err = objToMappedRange(view, pkg, result.Type.Object); err != nil {
			return nil, err
		}
	}
	return result, nil
}

func searchForEnclosing(pkg Package, path []ast.Node) types.Type {
	for _, n := range path {
		switch n := n.(type) {
		case *ast.SelectorExpr:
			if selection, ok := pkg.GetTypesInfo().Selections[n]; ok {
				return deref(selection.Recv())
			}
		case *ast.CompositeLit:
			if t, ok := pkg.GetTypesInfo().Types[n]; ok {
				return t.Type
			}
		case *ast.TypeSpec:
			if _, ok := n.Type.(*ast.StructType); ok {
				if t, ok := pkg.GetTypesInfo().Defs[n.Name]; ok {
					return t.Type()
				}
			}
		}
	}
	return nil
}

func typeToObject(typ types.Type) types.Object {
	switch typ := typ.(type) {
	case *types.Named:
		return typ.Obj()
	case *types.Pointer:
		return typeToObject(typ.Elem())
	default:
		return nil
	}
}

func hasErrorType(obj types.Object) bool {
	return types.IsInterface(obj.Type()) && obj.Pkg() == nil && obj.Name() == "error"
}

func objToNode(v View, pkg Package, obj types.Object) (ast.Decl, error) {
	declAST, _, err := findPosInPackage(v, pkg, obj.Pos())
	if err != nil {
		return nil, err
	}
	path, _ := astutil.PathEnclosingInterval(declAST, obj.Pos(), obj.Pos())
	if path == nil {
		return nil, errors.Errorf("no path for object %v", obj.Name())
	}
	for _, node := range path {
		switch node := node.(type) {
		case *ast.GenDecl:
			// Type names, fields, and methods.
			switch obj.(type) {
			case *types.TypeName, *types.Var, *types.Const, *types.Func:
				return node, nil
			}
		case *ast.FuncDecl:
			// Function signatures.
			if _, ok := obj.(*types.Func); ok {
				return node, nil
			}
		}
	}
	return nil, nil // didn't find a node, but don't fail
}

// importSpec handles positions inside of an *ast.ImportSpec.
func importSpec(s Snapshot, pkg Package, file *ast.File, pos token.Pos) (*IdentifierInfo, error) {
	var imp *ast.ImportSpec
	for _, spec := range file.Imports {
		if spec.Path.Pos() <= pos && pos < spec.Path.End() {
			imp = spec
		}
	}
	if imp == nil {
		return nil, nil
	}
	importPath, err := strconv.Unquote(imp.Path.Value)
	if err != nil {
		return nil, errors.Errorf("import path not quoted: %s (%v)", imp.Path.Value, err)
	}
	result := &IdentifierInfo{
		Snapshot: s,
		Name:     importPath,
		pkg:      pkg,
	}
	if result.mappedRange, err = posToMappedRange(s.View(), pkg, imp.Path.Pos(), imp.Path.End()); err != nil {
		return nil, err
	}
	// Consider the "declaration" of an import spec to be the imported package.
	importedPkg, err := pkg.GetImport(importPath)
	if err != nil {
		return nil, err
	}
	if importedPkg.GetSyntax() == nil {
		return nil, errors.Errorf("no syntax for for %q", importPath)
	}
	// Return all of the files in the package as the definition of the import spec.
	dest := pkg.GetSyntax()
	if len(dest) == 0 {
		return nil, errors.Errorf("package %q has no files", importPath)
	}

	for _, dst := range importedPkg.GetSyntax() {
		rng, err := posToMappedRange(s.View(), pkg, dst.Pos(), dst.End())
		if err != nil {
			return nil, err
		}
		result.Declaration.MappedRange = append(result.Declaration.MappedRange, rng)
	}

	result.Declaration.node = imp
	return result, nil
}

// typeSwitchImplicits returns all the implicit type switch objects
// that correspond to the leaf *ast.Ident.
func typeSwitchImplicits(pkg Package, path []ast.Node) []types.Object {
	ident, _ := path[0].(*ast.Ident)
	if ident == nil {
		return nil
	}

	var (
		ts     *ast.TypeSwitchStmt
		assign *ast.AssignStmt
		cc     *ast.CaseClause
		obj    = pkg.GetTypesInfo().ObjectOf(ident)
	)

	// Walk our ancestors to determine if our leaf ident refers to a
	// type switch variable, e.g. the "a" from "switch a := b.(type)".
Outer:
	for i := 1; i < len(path); i++ {
		switch n := path[i].(type) {
		case *ast.AssignStmt:
			// Check if ident is the "a" in "a := foo.(type)". The "a" in
			// this case has no types.Object, so check for ident equality.
			if len(n.Lhs) == 1 && n.Lhs[0] == ident {
				assign = n
			}
		case *ast.CaseClause:
			// Check if ident is a use of "a" within a case clause. Each
			// case clause implicitly maps "a" to a different types.Object,
			// so check if ident's object is the case clause's implicit
			// object.
			if obj != nil && pkg.GetTypesInfo().Implicits[n] == obj {
				cc = n
			}
		case *ast.TypeSwitchStmt:
			// Look for the type switch that owns our previously found
			// *ast.AssignStmt or *ast.CaseClause.

			if n.Assign == assign {
				ts = n
				break Outer
			}

			for _, stmt := range n.Body.List {
				if stmt == cc {
					ts = n
					break Outer
				}
			}
		}
	}

	if ts == nil {
		return nil
	}

	// Our leaf ident refers to a type switch variable. Fan out to the
	// type switch's implicit case clause objects.
	var objs []types.Object
	for _, cc := range ts.Body.List {
		if ccObj := pkg.GetTypesInfo().Implicits[cc]; ccObj != nil {
			objs = append(objs, ccObj)
		}
	}
	return objs
}
