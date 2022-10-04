// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package source

import (
	"bytes"
	"context"
	"fmt"
	"go/ast"
	"go/doc"
	"go/printer"
	"go/token"
	"go/types"
	"strings"

	"golang.org/x/tools/gopls/internal/lsp/protocol"
	"golang.org/x/tools/internal/event"
	"golang.org/x/tools/internal/event/tag"
	"golang.org/x/tools/internal/typeparams"
)

// FormatType returns the detail and kind for a types.Type.
func FormatType(typ types.Type, qf types.Qualifier) (detail string, kind protocol.CompletionItemKind) {
	if types.IsInterface(typ) {
		detail = "interface{...}"
		kind = protocol.InterfaceCompletion
	} else if _, ok := typ.(*types.Struct); ok {
		detail = "struct{...}"
		kind = protocol.StructCompletion
	} else if typ != typ.Underlying() {
		detail, kind = FormatType(typ.Underlying(), qf)
	} else {
		detail = types.TypeString(typ, qf)
		kind = protocol.ClassCompletion
	}
	return detail, kind
}

type signature struct {
	name, doc                   string
	typeParams, params, results []string
	variadic                    bool
	needResultParens            bool
}

func (s *signature) Format() string {
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

func (s *signature) TypeParams() []string {
	return s.typeParams
}

func (s *signature) Params() []string {
	return s.params
}

// NewBuiltinSignature returns signature for the builtin object with a given
// name, if a builtin object with the name exists.
func NewBuiltinSignature(ctx context.Context, s Snapshot, name string) (*signature, error) {
	builtin, err := s.BuiltinFile(ctx)
	if err != nil {
		return nil, err
	}
	obj := builtin.File.Scope.Lookup(name)
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
	params, _ := formatFieldList(ctx, s, decl.Type.Params, variadic)
	results, needResultParens := formatFieldList(ctx, s, decl.Type.Results, false)
	d := decl.Doc.Text()
	switch s.View().Options().HoverKind {
	case SynopsisDocumentation:
		d = doc.Synopsis(d)
	case NoDocumentation:
		d = ""
	}
	return &signature{
		doc:              d,
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

func formatFieldList(ctx context.Context, snapshot Snapshot, list *ast.FieldList, variadic bool) ([]string, bool) {
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
		if err := cfg.Fprint(b, snapshot.FileSet(), p.Type); err != nil {
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

// FormatTypeParams turns TypeParamList into its Go representation, such as:
// [T, Y]. Note that it does not print constraints as this is mainly used for
// formatting type params in method receivers.
func FormatTypeParams(tparams *typeparams.TypeParamList) string {
	if tparams == nil || tparams.Len() == 0 {
		return ""
	}
	var buf bytes.Buffer
	buf.WriteByte('[')
	for i := 0; i < tparams.Len(); i++ {
		if i > 0 {
			buf.WriteString(", ")
		}
		buf.WriteString(tparams.At(i).Obj().Name())
	}
	buf.WriteByte(']')
	return buf.String()
}

// NewSignature returns formatted signature for a types.Signature struct.
func NewSignature(ctx context.Context, s Snapshot, pkg Package, sig *types.Signature, comment *ast.CommentGroup, qf types.Qualifier) *signature {
	var tparams []string
	tpList := typeparams.ForSignature(sig)
	for i := 0; i < tpList.Len(); i++ {
		tparam := tpList.At(i)
		// TODO: is it possible to reuse the logic from FormatVarType here?
		s := tparam.Obj().Name() + " " + tparam.Constraint().String()
		tparams = append(tparams, s)
	}

	params := make([]string, 0, sig.Params().Len())
	for i := 0; i < sig.Params().Len(); i++ {
		el := sig.Params().At(i)
		typ := FormatVarType(s.FileSet(), pkg, el, qf)
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
		typ := FormatVarType(s.FileSet(), pkg, el, qf)
		if el.Name() == "" {
			results = append(results, typ)
		} else {
			if i == 0 {
				needResultParens = true
			}
			results = append(results, el.Name()+" "+typ)
		}
	}
	var d string
	if comment != nil {
		d = comment.Text()
	}
	switch s.View().Options().HoverKind {
	case SynopsisDocumentation:
		d = doc.Synopsis(d)
	case NoDocumentation:
		d = ""
	}
	return &signature{
		doc:              d,
		typeParams:       tparams,
		params:           params,
		results:          results,
		variadic:         sig.Variadic(),
		needResultParens: needResultParens,
	}
}

// FormatVarType formats a *types.Var, accounting for type aliases.
// To do this, it looks in the AST of the file in which the object is declared.
// On any errors, it always falls back to types.TypeString.
func FormatVarType(fset *token.FileSet, srcpkg Package, obj *types.Var, qf types.Qualifier) string {
	pkg, err := FindPackageFromPos(fset, srcpkg, obj.Pos())
	if err != nil {
		return types.TypeString(obj.Type(), qf)
	}

	_, field := FindDeclAndField(pkg.GetSyntax(), obj.Pos())
	if field == nil {
		return types.TypeString(obj.Type(), qf)
	}
	expr := field.Type

	// If the given expr refers to a type parameter, then use the
	// object's Type instead of the type parameter declaration. This helps
	// format the instantiated type as opposed to the original undeclared
	// generic type.
	if typeparams.IsTypeParam(pkg.GetTypesInfo().Types[expr].Type) {
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
	qualified = qualifyExpr(qualified, srcpkg, pkg, clonedInfo, qf)
	fmted := FormatNode(fset, qualified)
	return fmted
}

// qualifyExpr applies the "pkgName." prefix to any *ast.Ident in the expr.
func qualifyExpr(expr ast.Expr, srcpkg, pkg Package, clonedInfo map[token.Pos]*types.PkgName, qf types.Qualifier) ast.Expr {
	ast.Inspect(expr, func(n ast.Node) bool {
		switch n := n.(type) {
		case *ast.ArrayType, *ast.ChanType, *ast.Ellipsis,
			*ast.FuncType, *ast.MapType, *ast.ParenExpr,
			*ast.StarExpr, *ast.StructType, *ast.FieldList, *ast.Field:
			// These are the only types that are cloned by cloneExpr below,
			// so these are the only types that we can traverse and potentially
			// modify. This is not an ideal approach, but it works for now.

			// TODO(rFindley): can we eliminate this filtering entirely? This caused
			// bugs in the past (golang/go#50539)
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
			x.Name = qf(obj.Imported())
			return false
		case *ast.Ident:
			if srcpkg == pkg {
				return false
			}
			// Only add the qualifier if the identifier is exported.
			if ast.IsExported(n.Name) {
				pkgName := qf(pkg.GetTypes())
				n.Name = pkgName + "." + n.Name
			}
		}
		return false
	})
	return expr
}

// cloneExpr only clones expressions that appear in the parameters or return
// values of a function declaration. The original expression may be returned
// to the caller in 2 cases:
//
//  1. The expression has no pointer fields.
//  2. The expression cannot appear in an *ast.FuncType, making it
//     unnecessary to clone.
//
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
