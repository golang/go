// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements export filtering of an AST.

package doc

import (
	"go/ast"
	"go/token"
)

// filterIdentList removes unexported names from list in place
// and returns the resulting list.
func filterIdentList(list []*ast.Ident) []*ast.Ident {
	j := 0
	for _, x := range list {
		if token.IsExported(x.Name) {
			list[j] = x
			j++
		}
	}
	return list[0:j]
}

var underscore = ast.NewIdent("_")

func filterCompositeLit(lit *ast.CompositeLit, filter Filter, export bool) {
	n := len(lit.Elts)
	lit.Elts = filterExprList(lit.Elts, filter, export)
	if len(lit.Elts) < n {
		lit.Incomplete = true
	}
}

func filterExprList(list []ast.Expr, filter Filter, export bool) []ast.Expr {
	j := 0
	for _, exp := range list {
		switch x := exp.(type) {
		case *ast.CompositeLit:
			filterCompositeLit(x, filter, export)
		case *ast.KeyValueExpr:
			if x, ok := x.Key.(*ast.Ident); ok && !filter(x.Name) {
				continue
			}
			if x, ok := x.Value.(*ast.CompositeLit); ok {
				filterCompositeLit(x, filter, export)
			}
		}
		list[j] = exp
		j++
	}
	return list[0:j]
}

// updateIdentList replaces all unexported identifiers with underscore
// and reports whether at least one exported name exists.
func updateIdentList(list []*ast.Ident) (hasExported bool) {
	for i, x := range list {
		if token.IsExported(x.Name) {
			hasExported = true
		} else {
			list[i] = underscore
		}
	}
	return hasExported
}

// hasExportedName reports whether list contains any exported names.
func hasExportedName(list []*ast.Ident) bool {
	for _, x := range list {
		if x.IsExported() {
			return true
		}
	}
	return false
}

// removeAnonymousField removes anonymous fields named name from an interface.
func removeAnonymousField(name string, ityp *ast.InterfaceType) {
	list := ityp.Methods.List // we know that ityp.Methods != nil
	j := 0
	for _, field := range list {
		keepField := true
		if n := len(field.Names); n == 0 {
			// anonymous field
			if fname, _ := baseTypeName(field.Type); fname == name {
				keepField = false
			}
		}
		if keepField {
			list[j] = field
			j++
		}
	}
	if j < len(list) {
		ityp.Incomplete = true
	}
	ityp.Methods.List = list[0:j]
}

// filterFieldList removes unexported fields (field names) from the field list
// in place and reports whether fields were removed. Anonymous fields are
// recorded with the parent type. filterType is called with the types of
// all remaining fields.
func (r *reader) filterFieldList(parent *namedType, fields *ast.FieldList, ityp *ast.InterfaceType) (removedFields bool) {
	if fields == nil {
		return
	}
	list := fields.List
	j := 0
	for _, field := range list {
		keepField := false
		if n := len(field.Names); n == 0 {
			// anonymous field or embedded type or union element
			fname := r.recordAnonymousField(parent, field.Type)
			if fname != "" {
				if token.IsExported(fname) {
					keepField = true
				} else if ityp != nil && predeclaredTypes[fname] {
					// possibly an embedded predeclared type; keep it for now but
					// remember this interface so that it can be fixed if name is also
					// defined locally
					keepField = true
					r.remember(fname, ityp)
				}
			} else {
				// If we're operating on an interface, assume that this is an embedded
				// type or union element.
				//
				// TODO(rfindley): consider traversing into approximation/unions
				// elements to see if they are entirely unexported.
				keepField = ityp != nil
			}
		} else {
			field.Names = filterIdentList(field.Names)
			if len(field.Names) < n {
				removedFields = true
			}
			if len(field.Names) > 0 {
				keepField = true
			}
		}
		if keepField {
			r.filterType(nil, field.Type)
			list[j] = field
			j++
		}
	}
	if j < len(list) {
		removedFields = true
	}
	fields.List = list[0:j]
	return
}

// filterParamList applies filterType to each parameter type in fields.
func (r *reader) filterParamList(fields *ast.FieldList) {
	if fields != nil {
		for _, f := range fields.List {
			r.filterType(nil, f.Type)
		}
	}
}

// filterType strips any unexported struct fields or method types from typ
// in place. If fields (or methods) have been removed, the corresponding
// struct or interface type has the Incomplete field set to true.
func (r *reader) filterType(parent *namedType, typ ast.Expr) {
	switch t := typ.(type) {
	case *ast.Ident:
		// nothing to do
	case *ast.ParenExpr:
		r.filterType(nil, t.X)
	case *ast.StarExpr: // possibly an embedded type literal
		r.filterType(nil, t.X)
	case *ast.UnaryExpr:
		if t.Op == token.TILDE { // approximation element
			r.filterType(nil, t.X)
		}
	case *ast.BinaryExpr:
		if t.Op == token.OR { // union
			r.filterType(nil, t.X)
			r.filterType(nil, t.Y)
		}
	case *ast.ArrayType:
		r.filterType(nil, t.Elt)
	case *ast.StructType:
		if r.filterFieldList(parent, t.Fields, nil) {
			t.Incomplete = true
		}
	case *ast.FuncType:
		r.filterParamList(t.TypeParams)
		r.filterParamList(t.Params)
		r.filterParamList(t.Results)
	case *ast.InterfaceType:
		if r.filterFieldList(parent, t.Methods, t) {
			t.Incomplete = true
		}
	case *ast.MapType:
		r.filterType(nil, t.Key)
		r.filterType(nil, t.Value)
	case *ast.ChanType:
		r.filterType(nil, t.Value)
	}
}

func (r *reader) filterSpec(spec ast.Spec) bool {
	switch s := spec.(type) {
	case *ast.ImportSpec:
		// always keep imports so we can collect them
		return true
	case *ast.ValueSpec:
		s.Values = filterExprList(s.Values, token.IsExported, true)
		if len(s.Values) > 0 || s.Type == nil && len(s.Values) == 0 {
			// If there are values declared on RHS, just replace the unexported
			// identifiers on the LHS with underscore, so that it matches
			// the sequence of expression on the RHS.
			//
			// Similarly, if there are no type and values, then this expression
			// must be following an iota expression, where order matters.
			if updateIdentList(s.Names) {
				r.filterType(nil, s.Type)
				return true
			}
		} else {
			s.Names = filterIdentList(s.Names)
			if len(s.Names) > 0 {
				r.filterType(nil, s.Type)
				return true
			}
		}
	case *ast.TypeSpec:
		// Don't filter type parameters here, by analogy with function parameters
		// which are not filtered for top-level function declarations.
		if name := s.Name.Name; token.IsExported(name) {
			r.filterType(r.lookupType(s.Name.Name), s.Type)
			return true
		} else if IsPredeclared(name) {
			if r.shadowedPredecl == nil {
				r.shadowedPredecl = make(map[string]bool)
			}
			r.shadowedPredecl[name] = true
		}
	}
	return false
}

// copyConstType returns a copy of typ with position pos.
// typ must be a valid constant type.
// In practice, only (possibly qualified) identifiers are possible.
func copyConstType(typ ast.Expr, pos token.Pos) ast.Expr {
	switch typ := typ.(type) {
	case *ast.Ident:
		return &ast.Ident{Name: typ.Name, NamePos: pos}
	case *ast.SelectorExpr:
		if id, ok := typ.X.(*ast.Ident); ok {
			// presumably a qualified identifier
			return &ast.SelectorExpr{
				Sel: ast.NewIdent(typ.Sel.Name),
				X:   &ast.Ident{Name: id.Name, NamePos: pos},
			}
		}
	}
	return nil // shouldn't happen, but be conservative and don't panic
}

func (r *reader) filterSpecList(list []ast.Spec, tok token.Token) []ast.Spec {
	if tok == token.CONST {
		// Propagate any type information that would get lost otherwise
		// when unexported constants are filtered.
		var prevType ast.Expr
		for _, spec := range list {
			spec := spec.(*ast.ValueSpec)
			if spec.Type == nil && len(spec.Values) == 0 && prevType != nil {
				// provide current spec with an explicit type
				spec.Type = copyConstType(prevType, spec.Pos())
			}
			if hasExportedName(spec.Names) {
				// exported names are preserved so there's no need to propagate the type
				prevType = nil
			} else {
				prevType = spec.Type
			}
		}
	}

	j := 0
	for _, s := range list {
		if r.filterSpec(s) {
			list[j] = s
			j++
		}
	}
	return list[0:j]
}

func (r *reader) filterDecl(decl ast.Decl) bool {
	switch d := decl.(type) {
	case *ast.GenDecl:
		d.Specs = r.filterSpecList(d.Specs, d.Tok)
		return len(d.Specs) > 0
	case *ast.FuncDecl:
		// ok to filter these methods early because any
		// conflicting method will be filtered here, too -
		// thus, removing these methods early will not lead
		// to the false removal of possible conflicts
		return token.IsExported(d.Name.Name)
	}
	return false
}

// fileExports removes unexported declarations from src in place.
func (r *reader) fileExports(src *ast.File) {
	j := 0
	for _, d := range src.Decls {
		if r.filterDecl(d) {
			src.Decls[j] = d
			j++
		}
	}
	src.Decls = src.Decls[0:j]
}
