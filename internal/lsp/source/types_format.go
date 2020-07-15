// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"context"
	"fmt"
	"go/ast"
	"go/printer"
	"go/token"
	"go/types"
	"strings"

	"golang.org/x/tools/go/ast/astutil"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/lsp/debug/tag"
	"golang.org/x/tools/internal/lsp/protocol"
)

// formatType returns the detail and kind for a types.Type.
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

type signature struct {
	name, doc        string
	params, results  []string
	variadic         bool
	needResultParens bool
}

func (s *signature) format() string {
	var b strings.Builder
	b.WriteByte('(')
	for i, p := range s.params {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(p)
	}
	b.WriteByte(')')

	// Add space between parameters and results.
	if len(s.results) > 0 {
		b.WriteByte(' ')
	}
	if s.needResultParens {
		b.WriteByte('(')
	}
	for i, r := range s.results {
		if i > 0 {
			b.WriteString(", ")
		}
		b.WriteString(r)
	}
	if s.needResultParens {
		b.WriteByte(')')
	}
	return b.String()
}

func newBuiltinSignature(ctx context.Context, view View, name string) (*signature, error) {
	builtin, err := view.BuiltinPackage(ctx)
	if err != nil {
		return nil, err
	}
	obj := builtin.Package().Scope.Lookup(name)
	if obj == nil {
		return nil, fmt.Errorf("no builtin object for %s", name)
	}
	decl, ok := obj.Decl.(*ast.FuncDecl)
	if !ok {
		return nil, fmt.Errorf("no function declaration for builtin: %s", name)
	}
	if decl.Type == nil {
		return nil, fmt.Errorf("no type for builtin decl %s", decl.Name)
	}
	var variadic bool
	if decl.Type.Params.List != nil {
		numParams := len(decl.Type.Params.List)
		lastParam := decl.Type.Params.List[numParams-1]
		if _, ok := lastParam.Type.(*ast.Ellipsis); ok {
			variadic = true
		}
	}
	params, _ := formatFieldList(ctx, view, decl.Type.Params, variadic)
	results, needResultParens := formatFieldList(ctx, view, decl.Type.Results, false)
	return &signature{
		doc:              decl.Doc.Text(),
		name:             name,
		needResultParens: needResultParens,
		params:           params,
		results:          results,
		variadic:         variadic,
	}, nil
}

var replacer = strings.NewReplacer(
	`ComplexType`, `complex128`,
	`FloatType`, `float64`,
	`IntegerType`, `int`,
)

func formatFieldList(ctx context.Context, view View, list *ast.FieldList, variadic bool) ([]string, bool) {
	if list == nil {
		return nil, false
	}
	var writeResultParens bool
	var result []string
	for i := 0; i < len(list.List); i++ {
		if i >= 1 {
			writeResultParens = true
		}
		p := list.List[i]
		cfg := printer.Config{Mode: printer.UseSpaces | printer.TabIndent, Tabwidth: 4}
		b := &bytes.Buffer{}
		if err := cfg.Fprint(b, view.Session().Cache().FileSet(), p.Type); err != nil {
			event.Error(ctx, "unable to print type", nil, tag.Type.Of(p.Type))
			continue
		}
		typ := replacer.Replace(b.String())
		if len(p.Names) == 0 {
			result = append(result, typ)
		}
		for _, name := range p.Names {
			if name.Name != "" {
				if i == 0 {
					writeResultParens = true
				}
				result = append(result, fmt.Sprintf("%s %s", name.Name, typ))
			} else {
				result = append(result, typ)
			}
		}
	}
	if variadic {
		result[len(result)-1] = strings.Replace(result[len(result)-1], "[]", "...", 1)
	}
	return result, writeResultParens
}

func newSignature(ctx context.Context, s Snapshot, pkg Package, file *ast.File, name string, sig *types.Signature, comment *ast.CommentGroup, qf types.Qualifier) (*signature, error) {
	params := make([]string, 0, sig.Params().Len())
	for i := 0; i < sig.Params().Len(); i++ {
		el := sig.Params().At(i)
		typ := formatVarType(ctx, s, pkg, file, el, qf)
		p := typ
		if el.Name() != "" {
			p = el.Name() + " " + typ
		}
		params = append(params, p)
	}
	var needResultParens bool
	results := make([]string, 0, sig.Results().Len())
	for i := 0; i < sig.Results().Len(); i++ {
		if i >= 1 {
			needResultParens = true
		}
		el := sig.Results().At(i)
		typ := formatVarType(ctx, s, pkg, file, el, qf)
		if el.Name() == "" {
			results = append(results, typ)
		} else {
			if i == 0 {
				needResultParens = true
			}
			results = append(results, el.Name()+" "+typ)
		}
	}
	var doc string
	if comment != nil {
		doc = comment.Text()
	}
	return &signature{
		doc:              doc,
		params:           params,
		results:          results,
		variadic:         sig.Variadic(),
		needResultParens: needResultParens,
	}, nil
}

// formatVarType formats a *types.Var, accounting for type aliases.
// To do this, it looks in the AST of the file in which the object is declared.
// On any errors, it always fallbacks back to types.TypeString.
func formatVarType(ctx context.Context, s Snapshot, srcpkg Package, srcfile *ast.File, obj *types.Var, qf types.Qualifier) string {
	ph, pkg, err := findPosInPackage(s.View(), srcpkg, obj.Pos())
	if err != nil {
		return types.TypeString(obj.Type(), qf)
	}

	expr, err := varType(ctx, ph, obj)
	if err != nil {
		return types.TypeString(obj.Type(), qf)
	}

	// The type names in the AST may not be correctly qualified.
	// Determine the package name to use based on the package that originated
	// the query and the package in which the type is declared.
	// We then qualify the value by cloning the AST node and editing it.
	clonedInfo := make(map[token.Pos]*types.PkgName)
	qualified := cloneExpr(expr, pkg.GetTypesInfo(), clonedInfo)

	// If the request came from a different package than the one in which the
	// types are defined, we may need to modify the qualifiers.
	qualified = qualifyExpr(s.View().Session().Cache().FileSet(), qualified, srcpkg, pkg, srcfile, clonedInfo)
	fmted := formatNode(s.View().Session().Cache().FileSet(), qualified)
	return fmted
}

// varType returns the type expression for a *types.Var.
func varType(ctx context.Context, ph ParseGoHandle, obj *types.Var) (ast.Expr, error) {
	posToField, err := ph.PosToField(ctx)
	if err != nil {
		return nil, err
	}
	field := posToField[obj.Pos()]
	if field == nil {
		return nil, fmt.Errorf("no declaration for object %s", obj.Name())
	}
	typ, ok := field.Type.(ast.Expr)
	if !ok {
		return nil, fmt.Errorf("unexpected type for node (%T)", field.Type)
	}
	return typ, nil
}

// qualifyExpr applies the "pkgName." prefix to any *ast.Ident in the expr.
func qualifyExpr(fset *token.FileSet, expr ast.Expr, srcpkg, pkg Package, file *ast.File, clonedInfo map[token.Pos]*types.PkgName) ast.Expr {
	ast.Inspect(expr, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.ArrayType, *ast.ChanType, *ast.Ellipsis,
			*ast.FuncType, *ast.MapType, *ast.ParenExpr,
			*ast.StarExpr, *ast.StructType:
			// These are the only types that are cloned by cloneExpr below,
			// so these are the only types that we can traverse and potentially
			// modify. This is not an ideal approach, but it works for now.
			return true
		case *ast.SelectorExpr:
			// We may need to change any selectors in which the X is a package
			// name and the Sel is exported.
			x, ok := n.X.(*ast.Ident)
			if !ok {
				return false
			}
			obj, ok := clonedInfo[x.Pos()]
			if !ok {
				return false
			}
			pkgName := importedPkgName(fset, srcpkg.GetTypes(), obj.Imported(), file)
			if pkgName != "" {
				x.Name = pkgName
			}
			return false
		case *ast.Ident:
			if srcpkg == pkg {
				return false
			}
			// Only add the qualifier if the identifier is exported.
			if ast.IsExported(n.Name) {
				pkgName := importedPkgName(fset, srcpkg.GetTypes(), pkg.GetTypes(), file)
				n.Name = pkgName + "." + n.Name
			}
		}
		return false
	})
	return expr
}

// importedPkgName returns the package name used for pkg in srcpkg.
func importedPkgName(fset *token.FileSet, srcpkg, pkg *types.Package, file *ast.File) string {
	if srcpkg == pkg {
		return ""
	}
	// If the file already imports the package under another name, use that.
	for _, group := range astutil.Imports(fset, file) {
		for _, cand := range group {
			if strings.Trim(cand.Path.Value, `"`) == pkg.Path() {
				if cand.Name != nil && cand.Name.Name != "" {
					return cand.Name.Name
				}
			}
		}
	}
	return pkg.Name()
}

// cloneExpr only clones expressions that appear in the parameters or return
// values of a function declaration. The original expression may be returned
// to the caller in 2 cases:
//    (1) The expression has no pointer fields.
//    (2) The expression cannot appear in an *ast.FuncType, making it
//        unnecessary to clone.
// This function also keeps track of selector expressions in which the X is a
// package name and marks them in a map along with their type information, so
// that this information can be used when rewriting the expression.
//
// NOTE: This function is tailored to the use case of qualifyExpr, and should
// be used with caution.
func cloneExpr(expr ast.Expr, info *types.Info, clonedInfo map[token.Pos]*types.PkgName) ast.Expr {
	switch expr := expr.(type) {
	case *ast.ArrayType:
		return &ast.ArrayType{
			Lbrack: expr.Lbrack,
			Elt:    cloneExpr(expr.Elt, info, clonedInfo),
			Len:    expr.Len,
		}
	case *ast.ChanType:
		return &ast.ChanType{
			Arrow: expr.Arrow,
			Begin: expr.Begin,
			Dir:   expr.Dir,
			Value: cloneExpr(expr.Value, info, clonedInfo),
		}
	case *ast.Ellipsis:
		return &ast.Ellipsis{
			Ellipsis: expr.Ellipsis,
			Elt:      cloneExpr(expr.Elt, info, clonedInfo),
		}
	case *ast.FuncType:
		return &ast.FuncType{
			Func:    expr.Func,
			Params:  cloneFieldList(expr.Params, info, clonedInfo),
			Results: cloneFieldList(expr.Results, info, clonedInfo),
		}
	case *ast.Ident:
		return cloneIdent(expr)
	case *ast.MapType:
		return &ast.MapType{
			Map:   expr.Map,
			Key:   cloneExpr(expr.Key, info, clonedInfo),
			Value: cloneExpr(expr.Value, info, clonedInfo),
		}
	case *ast.ParenExpr:
		return &ast.ParenExpr{
			Lparen: expr.Lparen,
			Rparen: expr.Rparen,
			X:      cloneExpr(expr.X, info, clonedInfo),
		}
	case *ast.SelectorExpr:
		s := &ast.SelectorExpr{
			Sel: cloneIdent(expr.Sel),
			X:   cloneExpr(expr.X, info, clonedInfo),
		}
		if x, ok := expr.X.(*ast.Ident); ok && ast.IsExported(expr.Sel.Name) {
			if obj, ok := info.ObjectOf(x).(*types.PkgName); ok {
				clonedInfo[s.X.Pos()] = obj
			}
		}
		return s
	case *ast.StarExpr:
		return &ast.StarExpr{
			Star: expr.Star,
			X:    cloneExpr(expr.X, info, clonedInfo),
		}
	case *ast.StructType:
		return &ast.StructType{
			Struct:     expr.Struct,
			Fields:     cloneFieldList(expr.Fields, info, clonedInfo),
			Incomplete: expr.Incomplete,
		}
	default:
		return expr
	}
}

func cloneFieldList(fl *ast.FieldList, info *types.Info, clonedInfo map[token.Pos]*types.PkgName) *ast.FieldList {
	if fl == nil {
		return nil
	}
	if fl.List == nil {
		return &ast.FieldList{
			Closing: fl.Closing,
			Opening: fl.Opening,
		}
	}
	list := make([]*ast.Field, 0, len(fl.List))
	for _, f := range fl.List {
		var names []*ast.Ident
		for _, n := range f.Names {
			names = append(names, cloneIdent(n))
		}
		list = append(list, &ast.Field{
			Comment: f.Comment,
			Doc:     f.Doc,
			Names:   names,
			Tag:     f.Tag,
			Type:    cloneExpr(f.Type, info, clonedInfo),
		})
	}
	return &ast.FieldList{
		Closing: fl.Closing,
		Opening: fl.Opening,
		List:    list,
	}
}

func cloneIdent(ident *ast.Ident) *ast.Ident {
	return &ast.Ident{
		NamePos: ident.NamePos,
		Name:    ident.Name,
		Obj:     ident.Obj,
	}
}
