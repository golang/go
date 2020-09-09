// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package go2go

import (
	"fmt"
	"go/ast"
	"go/token"
	"go/types"
)

// typeArgs holds type arguments for the function that we are instantiating.
// We can look them up either with a types.Object associated with an ast.Ident,
// or with a types.TypeParam.
type typeArgs struct {
	types []types.Type // type arguments in order
	toAST map[types.Object]ast.Expr
	toTyp map[*types.TypeParam]types.Type
}

// newTypeArgs returns a new typeArgs value.
func newTypeArgs(typeTypes []types.Type) *typeArgs {
	return &typeArgs{
		types: typeTypes,
		toAST: make(map[types.Object]ast.Expr),
		toTyp: make(map[*types.TypeParam]types.Type),
	}
}

// typeArgsFromFields builds mappings from a list of type parameters
// expressed as ast.Field values.
func typeArgsFromFields(t *translator, astTypes []ast.Expr, typeTypes []types.Type, tparams []*ast.Field) *typeArgs {
	ta := newTypeArgs(typeTypes)
	i := 0
	for _, tf := range tparams {
		for _, tn := range tf.Names {
			obj, ok := t.importer.info.Defs[tn]
			if !ok {
				panic(fmt.Sprintf("no object for type parameter %q", tn))
			}
			objType := obj.Type()
			objParam, ok := objType.(*types.TypeParam)
			if !ok {
				panic(fmt.Sprintf("%v is not a TypeParam", objParam))
			}
			ta.add(obj, objParam, astTypes[i], typeTypes[i])
			i++
		}
	}
	return ta
}

// typeArgsFromExprs builds mappings from a list of type parameters
// expressed as ast.Expr values.
func typeArgsFromExprs(t *translator, astTypes []ast.Expr, typeTypes []types.Type, tparams []ast.Expr) *typeArgs {
	ta := newTypeArgs(typeTypes)
	for i, ti := range tparams {
		obj, ok := t.importer.info.Defs[ti.(*ast.Ident)]
		if !ok {
			panic(fmt.Sprintf("no object for type parameter %q", ti))
		}
		objType := obj.Type()
		objParam, ok := objType.(*types.TypeParam)
		if !ok {
			panic(fmt.Sprintf("%v is not a TypeParam", objParam))
		}
		ta.add(obj, objParam, astTypes[i], typeTypes[i])
	}
	return ta
}

// add adds mappings for obj to ast and typ.
func (ta *typeArgs) add(obj types.Object, objParam *types.TypeParam, ast ast.Expr, typ types.Type) {
	ta.toAST[obj] = ast
	ta.toTyp[objParam] = typ
}

// ast returns the AST for obj, and reports whether it exists.
func (ta *typeArgs) ast(obj types.Object) (ast.Expr, bool) {
	e, ok := ta.toAST[obj]
	return e, ok
}

// typ returns the Type for param, and reports whether it exists.
func (ta *typeArgs) typ(param *types.TypeParam) (types.Type, bool) {
	t, ok := ta.toTyp[param]
	return t, ok
}

// instantiateFunction creates a new instantiation of a function.
func (t *translator) instantiateFunction(qid qualifiedIdent, astTypes []ast.Expr, typeTypes []types.Type) (*ast.Ident, error) {
	name, err := t.instantiatedName(qid, typeTypes)
	if err != nil {
		return nil, err
	}

	decl, err := t.findFuncDecl(qid)
	if err != nil {
		return nil, err
	}

	ta := typeArgsFromFields(t, astTypes, typeTypes, decl.Type.TParams.List)

	instIdent := ast.NewIdent(name)

	newDecl := &ast.FuncDecl{
		Doc:  decl.Doc,
		Recv: t.instantiateFieldList(ta, decl.Recv),
		Name: instIdent,
		Type: t.instantiateExpr(ta, decl.Type).(*ast.FuncType),
		Body: t.instantiateBlockStmt(ta, decl.Body),
	}
	t.newDecls = append(t.newDecls, newDecl)

	return instIdent, nil
}

// findFuncDecl looks for the FuncDecl for qid.
func (t *translator) findFuncDecl(qid qualifiedIdent) (*ast.FuncDecl, error) {
	obj := t.findTypesObject(qid)
	if obj == nil {
		return nil, fmt.Errorf("could not find Object for %q", qid)
	}
	decl, ok := t.importer.lookupFunc(obj)
	if !ok {
		return nil, fmt.Errorf("could not find function body for %q", qid)
	}
	return decl, nil
}

// findTypesObject looks up the types.Object for qid.
// It returns nil if the ID is not found.
func (t *translator) findTypesObject(qid qualifiedIdent) types.Object {
	if qid.pkg == nil {
		if obj := t.importer.info.ObjectOf(qid.ident); obj != nil {
			// Ignore an embedded struct field.
			// We want the type, not the field.
			if _, ok := obj.(*types.Var); !ok {
				return obj
			}
		}
		return t.tpkg.Scope().Lookup(qid.ident.Name)
	} else {
		return qid.pkg.Scope().Lookup(qid.ident.Name)
	}
}

// instantiateType creates a new instantiation of a type.
func (t *translator) instantiateTypeDecl(qid qualifiedIdent, typ *types.Named, astTypes []ast.Expr, typeTypes []types.Type, instIdent *ast.Ident) (types.Type, error) {
	spec, err := t.findTypeSpec(qid)
	if err != nil {
		return nil, err
	}

	ta := typeArgsFromFields(t, astTypes, typeTypes, spec.TParams.List)

	newSpec := &ast.TypeSpec{
		Doc:     spec.Doc,
		Name:    instIdent,
		Assign:  spec.Assign,
		Type:    t.instantiateExpr(ta, spec.Type),
		Comment: spec.Comment,
	}
	newDecl := &ast.GenDecl{
		Tok:   token.TYPE,
		Specs: []ast.Spec{newSpec},
	}
	t.newDecls = append(t.newDecls, newDecl)

	// If typ already has type arguments, then they should be correct.
	// If it doesn't, we want to use typeTypes.
	typeWithTargs := typ
	if len(typ.TArgs()) == 0 {
		typeWithTargs = t.updateTArgs(typ, typeTypes)
	}

	instType := t.instantiateType(ta, typeWithTargs)

	t.setType(instIdent, instType)

	nm := typ.NumMethods()
	for i := 0; i < nm; i++ {
		method := typ.Method(i)
		mast, ok := t.importer.lookupFunc(method)
		if !ok {
			panic(fmt.Sprintf("no AST for method %v", method))
		}
		rtyp := mast.Recv.List[0].Type
		newRtype := ast.Expr(ast.NewIdent(instIdent.Name))
		if p, ok := rtyp.(*ast.StarExpr); ok {
			rtyp = p.X
			newRtype = &ast.StarExpr{
				X: newRtype,
			}
		}
		var tparams []ast.Expr
		switch rtyp := rtyp.(type) {
		case *ast.CallExpr:
			tparams = rtyp.Args
		case *ast.IndexExpr:
			tparams = []ast.Expr{rtyp.Index}
		default:
			panic("unexpected AST type")
		}
		ta := typeArgsFromExprs(t, astTypes, typeTypes, tparams)
		var names []*ast.Ident
		if mnames := mast.Recv.List[0].Names; len(mnames) > 0 {
			names = []*ast.Ident{mnames[0]}
		}
		newDecl := &ast.FuncDecl{
			Doc: mast.Doc,
			Recv: &ast.FieldList{
				Opening: mast.Recv.Opening,
				List: []*ast.Field{
					{
						Doc:     mast.Recv.List[0].Doc,
						Names:   names,
						Type:    newRtype,
						Comment: mast.Recv.List[0].Comment,
					},
				},
				Closing: mast.Recv.Closing,
			},
			Name: mast.Name,
			Type: t.instantiateExpr(ta, mast.Type).(*ast.FuncType),
			Body: t.instantiateBlockStmt(ta, mast.Body),
		}
		t.newDecls = append(t.newDecls, newDecl)
	}

	return instType, nil
}

// findTypeSpec looks for the TypeSpec for qid.
func (t *translator) findTypeSpec(qid qualifiedIdent) (*ast.TypeSpec, error) {
	obj := t.findTypesObject(qid)
	if obj == nil {
		return nil, fmt.Errorf("could not find Object for %q", qid)
	}
	spec, ok := t.importer.lookupTypeSpec(obj)
	if !ok {
		return nil, fmt.Errorf("could not find type spec for %q", qid)
	}

	if spec.Assign != token.NoPos {
		// This is a type alias that we need to resolve.
		typ := t.lookupType(spec.Type)
		if typ != nil {
			if named, ok := typ.(*types.Named); ok {
				if s, ok := t.importer.lookupTypeSpec(named.Obj()); ok {
					spec = s
				}
			}
		}
	}

	if spec.TParams == nil {
		return nil, fmt.Errorf("found type spec for %q but it has no type parameters", qid)
	}

	return spec, nil
}

// instantiateDecl instantiates a declaration.
func (t *translator) instantiateDecl(ta *typeArgs, d ast.Decl) ast.Decl {
	switch d := d.(type) {
	case nil:
		return nil
	case *ast.GenDecl:
		if len(d.Specs) == 0 {
			return d
		}
		nspecs := make([]ast.Spec, len(d.Specs))
		changed := false
		for i, s := range d.Specs {
			ns := t.instantiateSpec(ta, s)
			if ns != s {
				changed = true
			}
			nspecs[i] = ns
		}
		if !changed {
			return d
		}
		return &ast.GenDecl{
			Doc:    d.Doc,
			TokPos: d.TokPos,
			Tok:    d.Tok,
			Lparen: d.Lparen,
			Specs:  nspecs,
			Rparen: d.Rparen,
		}
	default:
		panic(fmt.Sprintf("unimplemented Decl %T", d))
	}
}

// instantiateSpec instantiates a spec node.
func (t *translator) instantiateSpec(ta *typeArgs, s ast.Spec) ast.Spec {
	switch s := s.(type) {
	case nil:
		return nil
	case *ast.ValueSpec:
		typ := t.instantiateExpr(ta, s.Type)
		values, changed := t.instantiateExprList(ta, s.Values)
		if typ == s.Type && !changed {
			return s
		}
		return &ast.ValueSpec{
			Doc:     s.Doc,
			Names:   s.Names,
			Type:    typ,
			Values:  values,
			Comment: s.Comment,
		}
	case *ast.TypeSpec:
		if s.TParams != nil {
			t.err = fmt.Errorf("%s: go2go tool does not support local parameterized types", t.fset.Position(s.Pos()))
			return nil
		}
		typ := t.instantiateExpr(ta, s.Type)
		if typ == s.Type {
			return s
		}
		return &ast.TypeSpec{
			Doc:     s.Doc,
			Name:    s.Name,
			Assign:  s.Assign,
			Type:    typ,
			Comment: s.Comment,
		}
	default:
		panic(fmt.Sprintf("unimplemented Spec %T", s))
	}
}

// instantiateStmt instantiates a statement.
func (t *translator) instantiateStmt(ta *typeArgs, s ast.Stmt) ast.Stmt {
	switch s := s.(type) {
	case nil:
		return nil
	case *ast.DeclStmt:
		decl := t.instantiateDecl(ta, s.Decl)
		if decl == s.Decl {
			return s
		}
		return &ast.DeclStmt{
			Decl: decl,
		}
	case *ast.EmptyStmt:
		return s
	case *ast.LabeledStmt:
		stmt := t.instantiateStmt(ta, s.Stmt)
		if stmt == s.Stmt {
			return s
		}
		return &ast.LabeledStmt{
			Label: s.Label,
			Colon: s.Colon,
			Stmt:  stmt,
		}
	case *ast.ExprStmt:
		x := t.instantiateExpr(ta, s.X)
		if x == s.X {
			return s
		}
		return &ast.ExprStmt{
			X: x,
		}
	case *ast.SendStmt:
		ch := t.instantiateExpr(ta, s.Chan)
		value := t.instantiateExpr(ta, s.Value)
		if ch == s.Chan && value == s.Value {
			return s
		}
		return &ast.SendStmt{
			Chan:  ch,
			Arrow: s.Arrow,
			Value: value,
		}
	case *ast.IncDecStmt:
		x := t.instantiateExpr(ta, s.X)
		if x == s.X {
			return s
		}
		return &ast.IncDecStmt{
			X:      x,
			TokPos: s.TokPos,
			Tok:    s.Tok,
		}
	case *ast.AssignStmt:
		lhs, lchanged := t.instantiateExprList(ta, s.Lhs)
		rhs, rchanged := t.instantiateExprList(ta, s.Rhs)
		if !lchanged && !rchanged {
			return s
		}
		return &ast.AssignStmt{
			Lhs:    lhs,
			TokPos: s.TokPos,
			Tok:    s.Tok,
			Rhs:    rhs,
		}
	case *ast.GoStmt:
		call := t.instantiateExpr(ta, s.Call).(*ast.CallExpr)
		if call == s.Call {
			return s
		}
		return &ast.GoStmt{
			Go:   s.Go,
			Call: call,
		}
	case *ast.DeferStmt:
		call := t.instantiateExpr(ta, s.Call).(*ast.CallExpr)
		if call == s.Call {
			return s
		}
		return &ast.DeferStmt{
			Defer: s.Defer,
			Call:  call,
		}
	case *ast.ReturnStmt:
		results, changed := t.instantiateExprList(ta, s.Results)
		if !changed {
			return s
		}
		return &ast.ReturnStmt{
			Return:  s.Return,
			Results: results,
		}
	case *ast.BranchStmt:
		return s
	case *ast.BlockStmt:
		return t.instantiateBlockStmt(ta, s)
	case *ast.IfStmt:
		init := t.instantiateStmt(ta, s.Init)
		cond := t.instantiateExpr(ta, s.Cond)
		body := t.instantiateBlockStmt(ta, s.Body)
		els := t.instantiateStmt(ta, s.Else)
		if init == s.Init && cond == s.Cond && body == s.Body && els == s.Else {
			return s
		}
		return &ast.IfStmt{
			If:   s.If,
			Init: init,
			Cond: cond,
			Body: body,
			Else: els,
		}
	case *ast.CaseClause:
		list, listChanged := t.instantiateExprList(ta, s.List)
		body, bodyChanged := t.instantiateStmtList(ta, s.Body)
		if !listChanged && !bodyChanged {
			return s
		}
		return &ast.CaseClause{
			Case:  s.Case,
			List:  list,
			Colon: s.Colon,
			Body:  body,
		}
	case *ast.SwitchStmt:
		init := t.instantiateStmt(ta, s.Init)
		tag := t.instantiateExpr(ta, s.Tag)
		body := t.instantiateBlockStmt(ta, s.Body)
		if init == s.Init && tag == s.Tag && body == s.Body {
			return s
		}
		return &ast.SwitchStmt{
			Switch: s.Switch,
			Init:   init,
			Tag:    tag,
			Body:   body,
		}
	case *ast.TypeSwitchStmt:
		init := t.instantiateStmt(ta, s.Init)
		assign := t.instantiateStmt(ta, s.Assign)
		body := t.instantiateBlockStmt(ta, s.Body)
		if init == s.Init && assign == s.Assign && body == s.Body {
			return s
		}
		return &ast.TypeSwitchStmt{
			Switch: s.Switch,
			Init:   init,
			Assign: assign,
			Body:   body,
		}
	case *ast.CommClause:
		comm := t.instantiateStmt(ta, s.Comm)
		body, bodyChanged := t.instantiateStmtList(ta, s.Body)
		if comm == s.Comm && !bodyChanged {
			return s
		}
		return &ast.CommClause{
			Case:  s.Case,
			Comm:  comm,
			Colon: s.Colon,
			Body:  body,
		}
	case *ast.SelectStmt:
		body := t.instantiateBlockStmt(ta, s.Body)
		if body == s.Body {
			return s
		}
		return &ast.SelectStmt{
			Select: s.Select,
			Body:   body,
		}
	case *ast.ForStmt:
		init := t.instantiateStmt(ta, s.Init)
		cond := t.instantiateExpr(ta, s.Cond)
		post := t.instantiateStmt(ta, s.Post)
		body := t.instantiateBlockStmt(ta, s.Body)
		if init == s.Init && cond == s.Cond && post == s.Post && body == s.Body {
			return s
		}
		return &ast.ForStmt{
			For:  s.For,
			Init: init,
			Cond: cond,
			Post: post,
			Body: body,
		}
	case *ast.RangeStmt:
		key := t.instantiateExpr(ta, s.Key)
		value := t.instantiateExpr(ta, s.Value)
		x := t.instantiateExpr(ta, s.X)
		body := t.instantiateBlockStmt(ta, s.Body)
		if key == s.Key && value == s.Value && x == s.X && body == s.Body {
			return s
		}
		return &ast.RangeStmt{
			For:    s.For,
			Key:    key,
			Value:  value,
			TokPos: s.TokPos,
			Tok:    s.Tok,
			X:      x,
			Body:   body,
		}
	default:
		panic(fmt.Sprintf("unimplemented Stmt %T", s))
	}
}

// instantiateBlockStmt instantiates a BlockStmt.
func (t *translator) instantiateBlockStmt(ta *typeArgs, pbs *ast.BlockStmt) *ast.BlockStmt {
	if pbs == nil {
		return nil
	}
	changed := false
	stmts := make([]ast.Stmt, len(pbs.List))
	for i, s := range pbs.List {
		is := t.instantiateStmt(ta, s)
		stmts[i] = is
		if is != s {
			changed = true
		}
	}
	if !changed {
		return pbs
	}
	return &ast.BlockStmt{
		Lbrace: pbs.Lbrace,
		List:   stmts,
		Rbrace: pbs.Rbrace,
	}
}

// instantiateStmtList instantiates a statement list.
func (t *translator) instantiateStmtList(ta *typeArgs, sl []ast.Stmt) ([]ast.Stmt, bool) {
	nsl := make([]ast.Stmt, len(sl))
	changed := false
	for i, s := range sl {
		ns := t.instantiateStmt(ta, s)
		if ns != s {
			changed = true
		}
		nsl[i] = ns
	}
	if !changed {
		return sl, false
	}
	return nsl, true
}

// instantiateFieldList instantiates a field list.
func (t *translator) instantiateFieldList(ta *typeArgs, fl *ast.FieldList) *ast.FieldList {
	if fl == nil {
		return nil
	}
	nfl := make([]*ast.Field, len(fl.List))
	changed := false
	for i, f := range fl.List {
		nf := t.instantiateField(ta, f)
		if nf != f {
			changed = true
		}
		nfl[i] = nf
	}
	if !changed {
		return fl
	}
	return &ast.FieldList{
		Opening: fl.Opening,
		List:    nfl,
		Closing: fl.Closing,
	}
}

// instantiateField instantiates a field.
func (t *translator) instantiateField(ta *typeArgs, f *ast.Field) *ast.Field {
	typ := t.instantiateExpr(ta, f.Type)
	if typ == f.Type {
		return f
	}
	return &ast.Field{
		Doc:     f.Doc,
		Names:   f.Names,
		Type:    typ,
		Tag:     f.Tag,
		Comment: f.Comment,
	}
}

// instantiateExpr instantiates an expression.
func (t *translator) instantiateExpr(ta *typeArgs, e ast.Expr) ast.Expr {
	var r ast.Expr
	switch e := e.(type) {
	case nil:
		return nil
	case *ast.Ident:
		obj := t.importer.info.ObjectOf(e)
		if obj != nil {
			if typ, ok := ta.ast(obj); ok {
				return typ
			}
		}
		return e
	case *ast.Ellipsis:
		elt := t.instantiateExpr(ta, e.Elt)
		if elt == e.Elt {
			return e
		}
		return &ast.Ellipsis{
			Ellipsis: e.Ellipsis,
			Elt:      elt,
		}
	case *ast.BasicLit:
		return e
	case *ast.FuncLit:
		typ := t.instantiateExpr(ta, e.Type).(*ast.FuncType)
		body := t.instantiateBlockStmt(ta, e.Body)
		if typ == e.Type && body == e.Body {
			return e
		}
		return &ast.FuncLit{
			Type: typ,
			Body: body,
		}
	case *ast.CompositeLit:
		typ := t.instantiateExpr(ta, e.Type)
		elts, changed := t.instantiateExprList(ta, e.Elts)
		if typ == e.Type && !changed {
			return e
		}
		return &ast.CompositeLit{
			Type:       typ,
			Lbrace:     e.Lbrace,
			Elts:       elts,
			Rbrace:     e.Rbrace,
			Incomplete: e.Incomplete,
		}
	case *ast.ParenExpr:
		x := t.instantiateExpr(ta, e.X)
		if x == e.X {
			return e
		}
		return &ast.ParenExpr{
			Lparen: e.Lparen,
			X:      x,
			Rparen: e.Rparen,
		}
	case *ast.SelectorExpr:
		x := t.instantiateExpr(ta, e.X)

		// If this is a reference to an instantiated embedded field,
		// we may need to instantiate it. The actual instantiation
		// is at the end of the function, as long as we create a
		// a new SelectorExpr when needed.
		instantiate := false
		obj := t.importer.info.ObjectOf(e.Sel)
		if obj != nil {
			if f, ok := obj.(*types.Var); ok && f.Embedded() {
				if named, ok := f.Type().(*types.Named); ok && len(named.TArgs()) > 0 && obj.Name() == named.Obj().Name() {
					instantiate = true
				}
			}
		}

		if x == e.X && !instantiate {
			return e
		}
		r = &ast.SelectorExpr{
			X:   x,
			Sel: e.Sel,
		}
	case *ast.IndexExpr:
		x := t.instantiateExpr(ta, e.X)
		index := t.instantiateExpr(ta, e.Index)
		origInferred, haveInferred := t.importer.info.Inferred[e]
		var newInferred types.Inferred
		inferredChanged := false
		if haveInferred {
			for _, typ := range origInferred.Targs {
				nt := t.instantiateType(ta, typ)
				newInferred.Targs = append(newInferred.Targs, nt)
				if nt != typ {
					inferredChanged = true
				}
			}
		}
		if x == e.X && index == e.Index && !inferredChanged {
			return e
		}
		r = &ast.IndexExpr{
			X:      x,
			Lbrack: e.Lbrack,
			Index:  index,
			Rbrack: e.Rbrack,
		}
		if haveInferred {
			t.importer.info.Inferred[r] = newInferred
		}
	case *ast.SliceExpr:
		x := t.instantiateExpr(ta, e.X)
		low := t.instantiateExpr(ta, e.Low)
		high := t.instantiateExpr(ta, e.High)
		max := t.instantiateExpr(ta, e.Max)
		if x == e.X && low == e.Low && high == e.High && max == e.Max {
			return e
		}
		r = &ast.SliceExpr{
			X:      x,
			Lbrack: e.Lbrack,
			Low:    low,
			High:   high,
			Max:    max,
			Slice3: e.Slice3,
			Rbrack: e.Rbrack,
		}
	case *ast.TypeAssertExpr:
		x := t.instantiateExpr(ta, e.X)
		typ := t.instantiateExpr(ta, e.Type)
		if x == e.X && typ == e.Type {
			return e
		}
		r = &ast.TypeAssertExpr{
			X:      x,
			Lparen: e.Lparen,
			Type:   typ,
			Rparen: e.Rparen,
		}
	case *ast.CallExpr:
		fun := t.instantiateExpr(ta, e.Fun)
		args, argsChanged := t.instantiateExprList(ta, e.Args)
		origInferred, haveInferred := t.importer.info.Inferred[e]
		var newInferred types.Inferred
		inferredChanged := false
		if haveInferred {
			for _, typ := range origInferred.Targs {
				nt := t.instantiateType(ta, typ)
				newInferred.Targs = append(newInferred.Targs, nt)
				if nt != typ {
					inferredChanged = true
				}
			}
			newInferred.Sig = t.instantiateType(ta, origInferred.Sig).(*types.Signature)
			if newInferred.Sig != origInferred.Sig {
				inferredChanged = true
			}
		}
		if fun == e.Fun && !argsChanged && !inferredChanged {
			return e
		}
		r = &ast.CallExpr{
			Fun:      fun,
			Lparen:   e.Lparen,
			Args:     args,
			Ellipsis: e.Ellipsis,
			Rparen:   e.Rparen,
		}
		if haveInferred {
			t.importer.info.Inferred[r] = newInferred
		}
	case *ast.StarExpr:
		x := t.instantiateExpr(ta, e.X)
		if x == e.X {
			return e
		}
		r = &ast.StarExpr{
			Star: e.Star,
			X:    x,
		}
	case *ast.UnaryExpr:
		x := t.instantiateExpr(ta, e.X)
		if x == e.X {
			return e
		}
		r = &ast.UnaryExpr{
			OpPos: e.OpPos,
			Op:    e.Op,
			X:     x,
		}
	case *ast.BinaryExpr:
		x := t.instantiateExpr(ta, e.X)
		y := t.instantiateExpr(ta, e.Y)
		if x == e.X && y == e.Y {
			return e
		}
		r = &ast.BinaryExpr{
			X:     x,
			OpPos: e.OpPos,
			Op:    e.Op,
			Y:     y,
		}
	case *ast.KeyValueExpr:
		key := t.instantiateExpr(ta, e.Key)
		value := t.instantiateExpr(ta, e.Value)
		if key == e.Key && value == e.Value {
			return e
		}
		r = &ast.KeyValueExpr{
			Key:   key,
			Colon: e.Colon,
			Value: value,
		}
	case *ast.ArrayType:
		ln := t.instantiateExpr(ta, e.Len)
		elt := t.instantiateExpr(ta, e.Elt)
		if ln == e.Len && elt == e.Elt {
			return e
		}
		r = &ast.ArrayType{
			Lbrack: e.Lbrack,
			Len:    ln,
			Elt:    elt,
		}
	case *ast.StructType:
		fields := e.Fields.List
		newFields := make([]*ast.Field, 0, len(fields))
		changed := false
		for _, f := range fields {
			newFields = append(newFields, f)
			if len(f.Names) > 0 {
				continue
			}
			isPtr := false
			ftyp := f.Type
			if star, ok := ftyp.(*ast.StarExpr); ok {
				ftyp = star.X
				isPtr = true
			}
			id, ok := ftyp.(*ast.Ident)
			if !ok {
				continue
			}
			typ := t.lookupType(id)
			if typ == nil {
				continue
			}
			typeParam, ok := typ.(*types.TypeParam)
			if !ok {
				continue
			}
			instType := t.instantiateType(ta, typeParam)
			if isPtr {
				instType = types.NewPointer(instType)
			}
			newField := &ast.Field{
				Doc:     f.Doc,
				Names:   []*ast.Ident{id},
				Type:    t.typeToAST(instType),
				Tag:     f.Tag,
				Comment: f.Comment,
			}
			newFields[len(newFields)-1] = newField
			changed = true
		}
		fl := e.Fields
		if changed {
			fl = &ast.FieldList{
				Opening: fl.Opening,
				List:    newFields,
				Closing: fl.Closing,
			}
		}
		newFl := t.instantiateFieldList(ta, fl)
		if !changed && newFl == fl {
			return e
		}
		r = &ast.StructType{
			Struct:     e.Struct,
			Fields:     newFl,
			Incomplete: e.Incomplete,
		}
	case *ast.FuncType:
		params := t.instantiateFieldList(ta, e.Params)
		results := t.instantiateFieldList(ta, e.Results)
		if e.TParams == nil && params == e.Params && results == e.Results {
			return e
		}
		r = &ast.FuncType{
			Func:    e.Func,
			TParams: nil,
			Params:  params,
			Results: results,
		}
	case *ast.InterfaceType:
		eMethods, eTypes := splitFieldList(e.Methods)
		methods := t.instantiateFieldList(ta, eMethods)
		types, typesChanged := t.instantiateExprList(ta, eTypes)
		if methods == e.Methods && !typesChanged {
			return e
		}
		r = &ast.InterfaceType{
			Interface:  e.Interface,
			Methods:    mergeFieldList(methods, types),
			Incomplete: e.Incomplete,
		}
	case *ast.MapType:
		key := t.instantiateExpr(ta, e.Key)
		value := t.instantiateExpr(ta, e.Value)
		if key == e.Key && value == e.Value {
			return e
		}
		r = &ast.MapType{
			Map:   e.Map,
			Key:   key,
			Value: value,
		}
	case *ast.ChanType:
		value := t.instantiateExpr(ta, e.Value)
		if value == e.Value {
			return e
		}
		r = &ast.ChanType{
			Begin: e.Begin,
			Arrow: e.Arrow,
			Dir:   e.Dir,
			Value: value,
		}
	default:
		panic(fmt.Sprintf("unimplemented Expr %T", e))
	}

	if et := t.lookupType(e); et != nil {
		t.setType(r, t.instantiateType(ta, et))
	}

	return r
}

// instantiateExprList instantiates an expression list.
func (t *translator) instantiateExprList(ta *typeArgs, el []ast.Expr) ([]ast.Expr, bool) {
	nel := make([]ast.Expr, len(el))
	changed := false
	for i, e := range el {
		ne := t.instantiateExpr(ta, e)
		if ne != e {
			changed = true
		}
		nel[i] = ne
	}
	if !changed {
		return el, false
	}
	return nel, true
}
