// Copyright 2011 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// TODO(rsc): Once there is better support for writing
// multi-package commands, this should really be in
// its own package, and then we can drop all the "reflect"
// prefixes on the global variables and functions.

package main

import (
	"go/ast"
	"go/token"
	"strings"
)

func init() {
	register(reflectFix)
}

var reflectFix = fix{
	"reflect",
	"2011-04-08",
	reflectFn,
	`Adapt code to new reflect API.

http://codereview.appspot.com/4281055
http://codereview.appspot.com/4433066
`,
}

// The reflect API change dropped the concrete types *reflect.ArrayType etc.
// Any type assertions prior to method calls can be deleted:
//	x.(*reflect.ArrayType).Len() -> x.Len()
//
// Any type checks can be replaced by assignment and check of Kind:
//	x, y := z.(*reflect.ArrayType)
// ->
//	x := z
//	y := x.Kind() == reflect.Array
//
// If z is an ordinary variable name and x is not subsequently assigned to,
// references to x can be replaced by z and the assignment deleted.
// We only bother if x and z are the same name.  
// If y is not subsequently assigned to and neither is x, references to
// y can be replaced by its expression.  We only bother when there is
// just one use or when the use appears in an if clause.
//
// Not all type checks result in a single Kind check.  The rewrite of the type check for
// reflect.ArrayOrSliceType checks x.Kind() against reflect.Array and reflect.Slice.
// The rewrite for *reflect.IntType checks againt Int, Int8, Int16, Int32, Int64.
// The rewrite for *reflect.UintType adds Uintptr.
//
// A type switch turns into an assignment and a switch on Kind:
//	switch x := y.(type) {
//	case reflect.ArrayOrSliceType:
//		...
//	case *reflect.ChanType:
//		...
//	case *reflect.IntType:
//		...
//	}
// ->
//	switch x := y; x.Kind() {
//	case reflect.Array, reflect.Slice:
//		...
//	case reflect.Chan:
//		...
//	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
//		...
//	}
//
// The same simplification applies: we drop x := x if x is not assigned
// to in the switch cases.
//
// Because the type check assignment includes a type assertion in its
// syntax and the rewrite traversal is bottom up, we must do a pass to
// rewrite the type check assignments and then a separate pass to 
// rewrite the type assertions.
//
// The same process applies to the API changes for reflect.Value.
//
// For both cases, but especially Value, the code needs to be aware
// of the type of a receiver when rewriting a method call.   For example,
// x.(*reflect.ArrayValue).Elem(i) becomes x.Index(i) while 
// x.(*reflect.MapValue).Elem(v) becomes x.MapIndex(v).
// In general, reflectFn needs to know the type of the receiver expression.
// In most cases (and in all the cases in the Go source tree), the toy
// type checker in typecheck.go provides enough information for gofix
// to make the rewrite.  If gofix misses a rewrite, the code that is left over
// will not compile, so it will be noticed immediately.

func reflectFn(f *ast.File) bool {
	if !imports(f, "reflect") {
		return false
	}

	fixed := false

	// Rewrite names in method calls.
	// Needs basic type information (see above).
	typeof, _ := typecheck(reflectTypeConfig, f)
	walk(f, func(n interface{}) {
		switch n := n.(type) {
		case *ast.SelectorExpr:
			typ := typeof[n.X]
			if m := reflectRewriteMethod[typ]; m != nil {
				if replace := m[n.Sel.Name]; replace != "" {
					n.Sel.Name = replace
					fixed = true
					return
				}
			}

			// For all reflect Values, replace SetValue with Set.
			if isReflectValue[typ] && n.Sel.Name == "SetValue" {
				n.Sel.Name = "Set"
				fixed = true
				return
			}

			// Replace reflect.MakeZero with reflect.Zero.
			if isPkgDot(n, "reflect", "MakeZero") {
				n.Sel.Name = "Zero"
				fixed = true
				return
			}
		}
	})

	// Replace PtrValue's PointTo(x) with Set(x.Addr()).
	walk(f, func(n interface{}) {
		call, ok := n.(*ast.CallExpr)
		if !ok || len(call.Args) != 1 {
			return
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || sel.Sel.Name != "PointTo" {
			return
		}
		typ := typeof[sel.X]
		if typ != "*reflect.PtrValue" {
			return
		}
		sel.Sel.Name = "Set"
		if !isTopName(call.Args[0], "nil") {
			call.Args[0] = &ast.SelectorExpr{
				X:   call.Args[0],
				Sel: ast.NewIdent("Addr()"),
			}
		}
		fixed = true
	})

	// Fix type switches.
	walk(f, func(n interface{}) {
		if reflectFixSwitch(n) {
			fixed = true
		}
	})

	// Fix type assertion checks (multiple assignment statements).
	// Have to work on the statement context (statement list or if statement)
	// so that we can insert an extra statement occasionally.
	// Ignoring for and switch because they don't come up in
	// typical code.
	walk(f, func(n interface{}) {
		switch n := n.(type) {
		case *[]ast.Stmt:
			// v is the replacement statement list.
			var v []ast.Stmt
			insert := func(x ast.Stmt) {
				v = append(v, x)
			}
			for i, x := range *n {
				// Tentatively append to v; if we rewrite x
				// we'll have to update the entry, so remember
				// the index.
				j := len(v)
				v = append(v, x)
				if reflectFixTypecheck(&x, insert, (*n)[i+1:]) {
					// reflectFixTypecheck may have overwritten x.
					// Update the entry we appended just before the call.
					v[j] = x
					fixed = true
				}
			}
			*n = v
		case *ast.IfStmt:
			x := &ast.ExprStmt{n.Cond}
			if reflectFixTypecheck(&n.Init, nil, []ast.Stmt{x, n.Body, n.Else}) {
				n.Cond = x.X
				fixed = true
			}
		}
	})

	// Warn about any typecheck statements that we missed.
	walk(f, reflectWarnTypecheckStmt)

	// Now that those are gone, fix remaining type assertions.
	// Delayed because the type checks have
	// type assertions as part of their syntax.
	walk(f, func(n interface{}) {
		if reflectFixAssert(n) {
			fixed = true
		}
	})

	// Now that the type assertions are gone, rewrite remaining
	// references to specific reflect types to use the general ones.
	walk(f, func(n interface{}) {
		ptr, ok := n.(*ast.Expr)
		if !ok {
			return
		}
		nn := *ptr
		typ := reflectType(nn)
		if typ == "" {
			return
		}
		if strings.HasSuffix(typ, "Type") {
			*ptr = newPkgDot(nn.Pos(), "reflect", "Type")
		} else {
			*ptr = newPkgDot(nn.Pos(), "reflect", "Value")
		}
		fixed = true
	})

	// Rewrite v.Set(nil) to v.Set(reflect.MakeZero(v.Type())).
	walk(f, func(n interface{}) {
		call, ok := n.(*ast.CallExpr)
		if !ok || len(call.Args) != 1 || !isTopName(call.Args[0], "nil") {
			return
		}
		sel, ok := call.Fun.(*ast.SelectorExpr)
		if !ok || !isReflectValue[typeof[sel.X]] || sel.Sel.Name != "Set" {
			return
		}
		call.Args[0] = &ast.CallExpr{
			Fun: newPkgDot(call.Args[0].Pos(), "reflect", "Zero"),
			Args: []ast.Expr{
				&ast.CallExpr{
					Fun: &ast.SelectorExpr{
						X:   sel.X,
						Sel: &ast.Ident{Name: "Type"},
					},
				},
			},
		}
		fixed = true
	})

	// Rewrite v != nil to v.IsValid().
	// Rewrite nil used as reflect.Value (in function argument or return) to reflect.Value{}.
	walk(f, func(n interface{}) {
		ptr, ok := n.(*ast.Expr)
		if !ok {
			return
		}
		if isTopName(*ptr, "nil") && isReflectValue[typeof[*ptr]] {
			*ptr = ast.NewIdent("reflect.Value{}")
			fixed = true
			return
		}
		nn, ok := (*ptr).(*ast.BinaryExpr)
		if !ok || (nn.Op != token.EQL && nn.Op != token.NEQ) || !isTopName(nn.Y, "nil") || !isReflectValue[typeof[nn.X]] {
			return
		}
		var call ast.Expr = &ast.CallExpr{
			Fun: &ast.SelectorExpr{
				X:   nn.X,
				Sel: &ast.Ident{Name: "IsValid"},
			},
		}
		if nn.Op == token.EQL {
			call = &ast.UnaryExpr{Op: token.NOT, X: call}
		}
		*ptr = call
		fixed = true
	})

	// Rewrite
	//	reflect.Typeof -> reflect.TypeOf,
	walk(f, func(n interface{}) {
		sel, ok := n.(*ast.SelectorExpr)
		if !ok {
			return
		}
		if isTopName(sel.X, "reflect") && sel.Sel.Name == "Typeof" {
			sel.Sel.Name = "TypeOf"
			fixed = true
		}
		if isTopName(sel.X, "reflect") && sel.Sel.Name == "NewValue" {
			sel.Sel.Name = "ValueOf"
			fixed = true
		}
	})

	return fixed
}

// reflectFixSwitch rewrites *n (if n is an *ast.Stmt) corresponding
// to a type switch.
func reflectFixSwitch(n interface{}) bool {
	ptr, ok := n.(*ast.Stmt)
	if !ok {
		return false
	}
	n = *ptr

	ts, ok := n.(*ast.TypeSwitchStmt)
	if !ok {
		return false
	}

	// Are any switch cases referring to reflect types?
	// (That is, is this an old reflect type switch?)
	for _, cas := range ts.Body.List {
		for _, typ := range cas.(*ast.CaseClause).List {
			if reflectType(typ) != "" {
				goto haveReflect
			}
		}
	}
	return false

haveReflect:
	// Now we know it's an old reflect type switch.  Prepare the new version,
	// but don't replace or edit the original until we're sure of success.

	// Figure out the initializer statement, if any, and the receiver for the Kind call.
	var init ast.Stmt
	var rcvr ast.Expr

	init = ts.Init
	switch n := ts.Assign.(type) {
	default:
		warn(ts.Pos(), "unexpected form in type switch")
		return false

	case *ast.AssignStmt:
		as := n
		ta := as.Rhs[0].(*ast.TypeAssertExpr)
		x := isIdent(as.Lhs[0])
		z := isIdent(ta.X)

		if isBlank(x) || x != nil && z != nil && x.Name == z.Name && !assignsTo(x, ts.Body.List) {
			// Can drop the variable creation.
			rcvr = ta.X
		} else {
			// Need to use initialization statement.
			if init != nil {
				warn(ts.Pos(), "cannot rewrite reflect type switch with initializing statement")
				return false
			}
			init = &ast.AssignStmt{
				Lhs:    []ast.Expr{as.Lhs[0]},
				TokPos: as.TokPos,
				Tok:    token.DEFINE,
				Rhs:    []ast.Expr{ta.X},
			}
			rcvr = as.Lhs[0]
		}

	case *ast.ExprStmt:
		rcvr = n.X.(*ast.TypeAssertExpr).X
	}

	// Prepare rewritten type switch (see large comment above for form).
	sw := &ast.SwitchStmt{
		Switch: ts.Switch,
		Init:   init,
		Tag: &ast.CallExpr{
			Fun: &ast.SelectorExpr{
				X: rcvr,
				Sel: &ast.Ident{
					NamePos: rcvr.End(),
					Name:    "Kind",
					Obj:     nil,
				},
			},
			Lparen: rcvr.End(),
			Rparen: rcvr.End(),
		},
		Body: &ast.BlockStmt{
			Lbrace: ts.Body.Lbrace,
			List:   nil, // to be filled in
			Rbrace: ts.Body.Rbrace,
		},
	}

	// Translate cases.
	for _, tcas := range ts.Body.List {
		tcas := tcas.(*ast.CaseClause)
		cas := &ast.CaseClause{
			Case:  tcas.Case,
			Colon: tcas.Colon,
			Body:  tcas.Body,
		}
		for _, t := range tcas.List {
			if isTopName(t, "nil") {
				cas.List = append(cas.List, newPkgDot(t.Pos(), "reflect", "Invalid"))
				continue
			}

			typ := reflectType(t)
			if typ == "" {
				warn(t.Pos(), "cannot rewrite reflect type switch case with non-reflect type %s", gofmt(t))
				cas.List = append(cas.List, t)
				continue
			}

			for _, k := range reflectKind[typ] {
				cas.List = append(cas.List, newPkgDot(t.Pos(), "reflect", k))
			}
		}
		sw.Body.List = append(sw.Body.List, cas)
	}

	// Everything worked.  Rewrite AST.
	*ptr = sw
	return true
}

// Rewrite x, y = z.(T) into
//	x = z
//	y = x.Kind() == K
// as described in the long comment above.
//
// If insert != nil, it can be called to insert a statement after *ptr in its block.
// If insert == nil, insertion is not possible.
// At most one call to insert is allowed.
//
// Scope gives the statements for which a declaration
// in *ptr would be in scope.
//
// The result is true of the statement was rewritten.
//
func reflectFixTypecheck(ptr *ast.Stmt, insert func(ast.Stmt), scope []ast.Stmt) bool {
	st := *ptr
	as, ok := st.(*ast.AssignStmt)
	if !ok || len(as.Lhs) != 2 || len(as.Rhs) != 1 {
		return false
	}

	ta, ok := as.Rhs[0].(*ast.TypeAssertExpr)
	if !ok {
		return false
	}
	typ := reflectType(ta.Type)
	if typ == "" {
		return false
	}

	// Have x, y := z.(t).
	x := isIdent(as.Lhs[0])
	y := isIdent(as.Lhs[1])
	z := isIdent(ta.X)

	// First step is x := z, unless it's x := x and the resulting x is never reassigned.
	// rcvr is the x in x.Kind().
	var rcvr ast.Expr
	if isBlank(x) ||
		as.Tok == token.DEFINE && x != nil && z != nil && x.Name == z.Name && !assignsTo(x, scope) {
		// Can drop the statement.
		// If we need to insert a statement later, now we have a slot.
		*ptr = &ast.EmptyStmt{}
		insert = func(x ast.Stmt) { *ptr = x }
		rcvr = ta.X
	} else {
		*ptr = &ast.AssignStmt{
			Lhs:    []ast.Expr{as.Lhs[0]},
			TokPos: as.TokPos,
			Tok:    as.Tok,
			Rhs:    []ast.Expr{ta.X},
		}
		rcvr = as.Lhs[0]
	}

	// Prepare x.Kind() == T expression appropriate to t.
	// If x is not a simple identifier, warn that we might be
	// reevaluating x.
	if x == nil {
		warn(as.Pos(), "rewrite reevaluates expr with possible side effects: %s", gofmt(as.Lhs[0]))
	}
	yExpr, yNotExpr := reflectKindEq(rcvr, reflectKind[typ])

	// Second step is y := x.Kind() == T, unless it's only used once
	// or we have no way to insert that statement.
	var yStmt *ast.AssignStmt
	if as.Tok == token.DEFINE && countUses(y, scope) <= 1 || insert == nil {
		// Can drop the statement and use the expression directly.
		rewriteUses(y,
			func(token.Pos) ast.Expr { return yExpr },
			func(token.Pos) ast.Expr { return yNotExpr },
			scope)
	} else {
		yStmt = &ast.AssignStmt{
			Lhs:    []ast.Expr{as.Lhs[1]},
			TokPos: as.End(),
			Tok:    as.Tok,
			Rhs:    []ast.Expr{yExpr},
		}
		insert(yStmt)
	}
	return true
}

// reflectKindEq returns the expression z.Kind() == kinds[0] || z.Kind() == kinds[1] || ...
// and its negation.
// The qualifier "reflect." is inserted before each kinds[i] expression.
func reflectKindEq(z ast.Expr, kinds []string) (ast.Expr, ast.Expr) {
	n := len(kinds)
	if n == 1 {
		y := &ast.BinaryExpr{
			X: &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   z,
					Sel: ast.NewIdent("Kind"),
				},
			},
			Op: token.EQL,
			Y:  newPkgDot(token.NoPos, "reflect", kinds[0]),
		}
		ynot := &ast.BinaryExpr{
			X: &ast.CallExpr{
				Fun: &ast.SelectorExpr{
					X:   z,
					Sel: ast.NewIdent("Kind"),
				},
			},
			Op: token.NEQ,
			Y:  newPkgDot(token.NoPos, "reflect", kinds[0]),
		}
		return y, ynot
	}

	x, xnot := reflectKindEq(z, kinds[0:n-1])
	y, ynot := reflectKindEq(z, kinds[n-1:])

	or := &ast.BinaryExpr{
		X:  x,
		Op: token.LOR,
		Y:  y,
	}
	andnot := &ast.BinaryExpr{
		X:  xnot,
		Op: token.LAND,
		Y:  ynot,
	}
	return or, andnot
}

// if x represents a known old reflect type/value like *reflect.PtrType or reflect.ArrayOrSliceValue,
// reflectType returns the string form of that type.
func reflectType(x ast.Expr) string {
	ptr, ok := x.(*ast.StarExpr)
	if ok {
		x = ptr.X
	}

	sel, ok := x.(*ast.SelectorExpr)
	if !ok || !isName(sel.X, "reflect") {
		return ""
	}

	var s = "reflect."
	if ptr != nil {
		s = "*reflect."
	}
	s += sel.Sel.Name

	if reflectKind[s] != nil {
		return s
	}
	return ""
}

// reflectWarnTypecheckStmt warns about statements
// of the form x, y = z.(T) for any old reflect type T.
// The last pass should have gotten them all, and if it didn't,
// the next pass is going to turn them into x, y = z.
func reflectWarnTypecheckStmt(n interface{}) {
	as, ok := n.(*ast.AssignStmt)
	if !ok || len(as.Lhs) != 2 || len(as.Rhs) != 1 {
		return
	}
	ta, ok := as.Rhs[0].(*ast.TypeAssertExpr)
	if !ok || reflectType(ta.Type) == "" {
		return
	}
	warn(n.(ast.Node).Pos(), "unfixed reflect type check")
}

// reflectFixAssert rewrites x.(T) to x for any old reflect type T.
func reflectFixAssert(n interface{}) bool {
	ptr, ok := n.(*ast.Expr)
	if ok {
		ta, ok := (*ptr).(*ast.TypeAssertExpr)
		if ok && reflectType(ta.Type) != "" {
			*ptr = ta.X
			return true
		}
	}
	return false
}

// Tables describing the transformations.

// Description of old reflect API for partial type checking.
// We pretend the Elem method is on Type and Value instead
// of enumerating all the types it is actually on.
// Also, we pretend that ArrayType etc embeds Type for the
// purposes of describing the API.  (In fact they embed commonType,
// which implements Type.)
var reflectTypeConfig = &TypeConfig{
	Type: map[string]*Type{
		"reflect.ArrayOrSliceType":  {Embed: []string{"reflect.Type"}},
		"reflect.ArrayOrSliceValue": {Embed: []string{"reflect.Value"}},
		"reflect.ArrayType":         {Embed: []string{"reflect.Type"}},
		"reflect.ArrayValue":        {Embed: []string{"reflect.Value"}},
		"reflect.BoolType":          {Embed: []string{"reflect.Type"}},
		"reflect.BoolValue":         {Embed: []string{"reflect.Value"}},
		"reflect.ChanType":          {Embed: []string{"reflect.Type"}},
		"reflect.ChanValue": {
			Method: map[string]string{
				"Recv":    "func() (reflect.Value, bool)",
				"TryRecv": "func() (reflect.Value, bool)",
			},
			Embed: []string{"reflect.Value"},
		},
		"reflect.ComplexType":  {Embed: []string{"reflect.Type"}},
		"reflect.ComplexValue": {Embed: []string{"reflect.Value"}},
		"reflect.FloatType":    {Embed: []string{"reflect.Type"}},
		"reflect.FloatValue":   {Embed: []string{"reflect.Value"}},
		"reflect.FuncType": {
			Method: map[string]string{
				"In":  "func(int) reflect.Type",
				"Out": "func(int) reflect.Type",
			},
			Embed: []string{"reflect.Type"},
		},
		"reflect.FuncValue": {
			Method: map[string]string{
				"Call": "func([]reflect.Value) []reflect.Value",
			},
		},
		"reflect.IntType":        {Embed: []string{"reflect.Type"}},
		"reflect.IntValue":       {Embed: []string{"reflect.Value"}},
		"reflect.InterfaceType":  {Embed: []string{"reflect.Type"}},
		"reflect.InterfaceValue": {Embed: []string{"reflect.Value"}},
		"reflect.MapType": {
			Method: map[string]string{
				"Key": "func() reflect.Type",
			},
			Embed: []string{"reflect.Type"},
		},
		"reflect.MapValue": {
			Method: map[string]string{
				"Keys": "func() []reflect.Value",
			},
			Embed: []string{"reflect.Value"},
		},
		"reflect.Method": {
			Field: map[string]string{
				"Type": "*reflect.FuncType",
				"Func": "*reflect.FuncValue",
			},
		},
		"reflect.PtrType":   {Embed: []string{"reflect.Type"}},
		"reflect.PtrValue":  {Embed: []string{"reflect.Value"}},
		"reflect.SliceType": {Embed: []string{"reflect.Type"}},
		"reflect.SliceValue": {
			Method: map[string]string{
				"Slice": "func(int, int) *reflect.SliceValue",
			},
			Embed: []string{"reflect.Value"},
		},
		"reflect.StringType":  {Embed: []string{"reflect.Type"}},
		"reflect.StringValue": {Embed: []string{"reflect.Value"}},
		"reflect.StructField": {
			Field: map[string]string{
				"Type": "reflect.Type",
			},
		},
		"reflect.StructType": {
			Method: map[string]string{
				"Field":           "func() reflect.StructField",
				"FieldByIndex":    "func() reflect.StructField",
				"FieldByName":     "func() reflect.StructField,bool",
				"FieldByNameFunc": "func() reflect.StructField,bool",
			},
			Embed: []string{"reflect.Type"},
		},
		"reflect.StructValue": {
			Method: map[string]string{
				"Field":           "func() reflect.Value",
				"FieldByIndex":    "func() reflect.Value",
				"FieldByName":     "func() reflect.Value",
				"FieldByNameFunc": "func() reflect.Value",
			},
			Embed: []string{"reflect.Value"},
		},
		"reflect.Type": {
			Method: map[string]string{
				"Elem":   "func() reflect.Type",
				"Method": "func() reflect.Method",
			},
		},
		"reflect.UintType":           {Embed: []string{"reflect.Type"}},
		"reflect.UintValue":          {Embed: []string{"reflect.Value"}},
		"reflect.UnsafePointerType":  {Embed: []string{"reflect.Type"}},
		"reflect.UnsafePointerValue": {Embed: []string{"reflect.Value"}},
		"reflect.Value": {
			Method: map[string]string{
				"Addr":     "func() *reflect.PtrValue",
				"Elem":     "func() reflect.Value",
				"Method":   "func() *reflect.FuncValue",
				"SetValue": "func(reflect.Value)",
			},
		},
	},
	Func: map[string]string{
		"reflect.Append":      "*reflect.SliceValue",
		"reflect.AppendSlice": "*reflect.SliceValue",
		"reflect.Indirect":    "reflect.Value",
		"reflect.MakeSlice":   "*reflect.SliceValue",
		"reflect.MakeChan":    "*reflect.ChanValue",
		"reflect.MakeMap":     "*reflect.MapValue",
		"reflect.MakeZero":    "reflect.Value",
		"reflect.NewValue":    "reflect.Value",
		"reflect.PtrTo":       "*reflect.PtrType",
		"reflect.Typeof":      "reflect.Type",
	},
}

var reflectRewriteMethod = map[string]map[string]string{
	// The type API didn't change much.
	"*reflect.ChanType": {"Dir": "ChanDir"},
	"*reflect.FuncType": {"DotDotDot": "IsVariadic"},

	// The value API has longer names to disambiguate
	// methods with different signatures.
	"reflect.ArrayOrSliceValue": { // interface, not pointer
		"Elem": "Index",
	},
	"*reflect.ArrayValue": {
		"Elem": "Index",
	},
	"*reflect.BoolValue": {
		"Get": "Bool",
		"Set": "SetBool",
	},
	"*reflect.ChanValue": {
		"Get": "Pointer",
	},
	"*reflect.ComplexValue": {
		"Get":      "Complex",
		"Set":      "SetComplex",
		"Overflow": "OverflowComplex",
	},
	"*reflect.FloatValue": {
		"Get":      "Float",
		"Set":      "SetFloat",
		"Overflow": "OverflowFloat",
	},
	"*reflect.FuncValue": {
		"Get": "Pointer",
	},
	"*reflect.IntValue": {
		"Get":      "Int",
		"Set":      "SetInt",
		"Overflow": "OverflowInt",
	},
	"*reflect.InterfaceValue": {
		"Get": "InterfaceData",
	},
	"*reflect.MapValue": {
		"Elem":    "MapIndex",
		"Get":     "Pointer",
		"Keys":    "MapKeys",
		"SetElem": "SetMapIndex",
	},
	"*reflect.PtrValue": {
		"Get": "Pointer",
	},
	"*reflect.SliceValue": {
		"Elem": "Index",
		"Get":  "Pointer",
	},
	"*reflect.StringValue": {
		"Get": "String",
		"Set": "SetString",
	},
	"*reflect.UintValue": {
		"Get":      "Uint",
		"Set":      "SetUint",
		"Overflow": "OverflowUint",
	},
	"*reflect.UnsafePointerValue": {
		"Get": "Pointer",
		"Set": "SetPointer",
	},
}

var reflectKind = map[string][]string{
	"reflect.ArrayOrSliceType":   {"Array", "Slice"}, // interface, not pointer
	"*reflect.ArrayType":         {"Array"},
	"*reflect.BoolType":          {"Bool"},
	"*reflect.ChanType":          {"Chan"},
	"*reflect.ComplexType":       {"Complex64", "Complex128"},
	"*reflect.FloatType":         {"Float32", "Float64"},
	"*reflect.FuncType":          {"Func"},
	"*reflect.IntType":           {"Int", "Int8", "Int16", "Int32", "Int64"},
	"*reflect.InterfaceType":     {"Interface"},
	"*reflect.MapType":           {"Map"},
	"*reflect.PtrType":           {"Ptr"},
	"*reflect.SliceType":         {"Slice"},
	"*reflect.StringType":        {"String"},
	"*reflect.StructType":        {"Struct"},
	"*reflect.UintType":          {"Uint", "Uint8", "Uint16", "Uint32", "Uint64", "Uintptr"},
	"*reflect.UnsafePointerType": {"UnsafePointer"},

	"reflect.ArrayOrSliceValue":   {"Array", "Slice"}, // interface, not pointer
	"*reflect.ArrayValue":         {"Array"},
	"*reflect.BoolValue":          {"Bool"},
	"*reflect.ChanValue":          {"Chan"},
	"*reflect.ComplexValue":       {"Complex64", "Complex128"},
	"*reflect.FloatValue":         {"Float32", "Float64"},
	"*reflect.FuncValue":          {"Func"},
	"*reflect.IntValue":           {"Int", "Int8", "Int16", "Int32", "Int64"},
	"*reflect.InterfaceValue":     {"Interface"},
	"*reflect.MapValue":           {"Map"},
	"*reflect.PtrValue":           {"Ptr"},
	"*reflect.SliceValue":         {"Slice"},
	"*reflect.StringValue":        {"String"},
	"*reflect.StructValue":        {"Struct"},
	"*reflect.UintValue":          {"Uint", "Uint8", "Uint16", "Uint32", "Uint64", "Uintptr"},
	"*reflect.UnsafePointerValue": {"UnsafePointer"},
}

var isReflectValue = map[string]bool{
	"reflect.ArrayOrSliceValue":   true, // interface, not pointer
	"*reflect.ArrayValue":         true,
	"*reflect.BoolValue":          true,
	"*reflect.ChanValue":          true,
	"*reflect.ComplexValue":       true,
	"*reflect.FloatValue":         true,
	"*reflect.FuncValue":          true,
	"*reflect.IntValue":           true,
	"*reflect.InterfaceValue":     true,
	"*reflect.MapValue":           true,
	"*reflect.PtrValue":           true,
	"*reflect.SliceValue":         true,
	"*reflect.StringValue":        true,
	"*reflect.StructValue":        true,
	"*reflect.UintValue":          true,
	"*reflect.UnsafePointerValue": true,
	"reflect.Value":               true, // interface, not pointer
}
