// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements typechecking of expressions.

package types

import (
	"go/ast"
	"go/token"
	"strconv"
)

// TODO(gri) Cleanups
// - don't print error messages referring to invalid types (they are likely spurious errors)
// - simplify invalid handling: maybe just use Typ[Invalid] as marker, get rid of invalid Mode for values?
// - rethink error handling: should all callers check if x.mode == valid after making a call?
// - at the moment, iota is passed around almost everywhere - in many places we know it cannot be used

// TODO(gri) API issues
// - clients need access to result type information (tuples)
// - clients need access to constant values
// - clients need access to built-in type information

// TODO(gri) Bugs
// - expression hints are (correctly) used untyped for composite literal components, but also
//   in possibly overlapping use as hints for shift expressions - investigate

func (check *checker) collectParams(list *ast.FieldList, variadicOk bool) (params ObjList, isVariadic bool) {
	if list == nil {
		return
	}
	var last *ast.Object
	for i, field := range list.List {
		ftype := field.Type
		if t, _ := ftype.(*ast.Ellipsis); t != nil {
			ftype = t.Elt
			if variadicOk && i == len(list.List)-1 {
				isVariadic = true
			} else {
				check.invalidAST(field.Pos(), "... not permitted")
				// ok to continue
			}
		}
		// the parser ensures that f.Tag is nil and we don't
		// care if a constructed AST contains a non-nil tag
		typ := check.typ(ftype, true)
		if len(field.Names) > 0 {
			// named parameter
			for _, name := range field.Names {
				obj := name.Obj
				obj.Type = typ
				params = append(params, obj)
				last = obj
			}
		} else {
			// anonymous parameter
			obj := ast.NewObj(ast.Var, "")
			obj.Type = typ
			params = append(params, obj)
			last = obj
		}
	}
	// For a variadic function, change the last parameter's object type
	// from T to []T (this is the type used inside the function), but
	// keep a copy of the object with the original type T in the params
	// list (this is the externally visible type).
	if isVariadic {
		// if isVariadic is set, last must exist and len(params) > 0
		copy := *last
		last.Type = &Slice{Elt: last.Type.(Type)}
		params[len(params)-1] = &copy
	}
	return
}

func (check *checker) collectMethods(list *ast.FieldList) (methods ObjList) {
	if list == nil {
		return
	}
	for _, f := range list.List {
		typ := check.typ(f.Type, len(f.Names) > 0) // cycles are not ok for embedded interfaces
		// the parser ensures that f.Tag is nil and we don't
		// care if a constructed AST contains a non-nil tag
		if len(f.Names) > 0 {
			// methods (the parser ensures that there's only one
			// and we don't care if a constructed AST has more)
			if _, ok := typ.(*Signature); !ok {
				check.invalidAST(f.Type.Pos(), "%s is not a method signature", typ)
				continue
			}
			for _, name := range f.Names {
				obj := name.Obj
				obj.Type = typ
				methods = append(methods, obj)
			}
		} else {
			// embedded interface
			utyp := underlying(typ)
			if ityp, ok := utyp.(*Interface); ok {
				methods = append(methods, ityp.Methods...)
			} else if utyp != Typ[Invalid] {
				// if utyp is invalid, don't complain (the root cause was reported before)
				check.errorf(f.Type.Pos(), "%s is not an interface type", typ)
			}
		}
	}
	// check for double declarations
	methods.Sort()
	prev := ""
	for _, obj := range methods {
		if obj.Name == prev {
			check.errorf(list.Pos(), "multiple methods named %s", prev)
			return // keep multiple entries, lookup will only return the first entry
		}
	}
	return
}

func (check *checker) tag(t *ast.BasicLit) string {
	if t != nil {
		if t.Kind == token.STRING {
			if val, err := strconv.Unquote(t.Value); err == nil {
				return val
			}
		}
		check.invalidAST(t.Pos(), "incorrect tag syntax: %q", t.Value)
	}
	return ""
}

func (check *checker) collectFields(list *ast.FieldList, cycleOk bool) (fields []*StructField) {
	if list == nil {
		return
	}
	for _, f := range list.List {
		typ := check.typ(f.Type, cycleOk)
		tag := check.tag(f.Tag)
		if len(f.Names) > 0 {
			// named fields
			for _, name := range f.Names {
				fields = append(fields, &StructField{name.Name, typ, tag, false})
			}
		} else {
			// anonymous field
			switch t := deref(typ).(type) {
			case *Basic:
				fields = append(fields, &StructField{t.Name, typ, tag, true})
			case *NamedType:
				fields = append(fields, &StructField{t.Obj.Name, typ, tag, true})
			default:
				if typ != Typ[Invalid] {
					check.invalidAST(f.Type.Pos(), "anonymous field type %s must be named", typ)
				}
			}
		}
	}
	return
}

type opPredicates map[token.Token]func(Type) bool

var unaryOpPredicates = opPredicates{
	token.ADD: isNumeric,
	token.SUB: isNumeric,
	token.XOR: isInteger,
	token.NOT: isBoolean,
}

func (check *checker) op(m opPredicates, x *operand, op token.Token) bool {
	if pred := m[op]; pred != nil {
		if !pred(x.typ) {
			check.invalidOp(x.pos(), "operator %s not defined for %s", op, x)
			return false
		}
	} else {
		check.invalidAST(x.pos(), "unknown operator %s", op)
		return false
	}
	return true
}

func (check *checker) unary(x *operand, op token.Token) {
	switch op {
	case token.AND:
		// TODO(gri) need to check for composite literals, somehow (they are not variables, in general)
		if x.mode != variable {
			check.invalidOp(x.pos(), "cannot take address of %s", x)
			goto Error
		}
		x.typ = &Pointer{Base: x.typ}
		return

	case token.ARROW:
		typ, ok := underlying(x.typ).(*Chan)
		if !ok {
			check.invalidOp(x.pos(), "cannot receive from non-channel %s", x)
			goto Error
		}
		if typ.Dir&ast.RECV == 0 {
			check.invalidOp(x.pos(), "cannot receive from send-only channel %s", x)
			goto Error
		}
		x.mode = valueok
		x.typ = typ.Elt
		return
	}

	if !check.op(unaryOpPredicates, x, op) {
		goto Error
	}

	if x.mode == constant {
		switch op {
		case token.ADD:
			// nothing to do
		case token.SUB:
			x.val = binaryOpConst(zeroConst, x.val, token.SUB, false)
		case token.XOR:
			x.val = binaryOpConst(minusOneConst, x.val, token.XOR, false)
		case token.NOT:
			x.val = !x.val.(bool)
		default:
			unreachable() // operators where checked by check.op
		}
		// Typed constants must be representable in
		// their type after each constant operation.
		check.isRepresentable(x, underlying(x.typ).(*Basic))
		return
	}

	x.mode = value
	// x.typ remains unchanged
	return

Error:
	x.mode = invalid
}

func isShift(op token.Token) bool {
	return op == token.SHL || op == token.SHR
}

func isComparison(op token.Token) bool {
	// Note: tokens are not ordered well to make this much easier
	switch op {
	case token.EQL, token.NEQ, token.LSS, token.LEQ, token.GTR, token.GEQ:
		return true
	}
	return false
}

// isRepresentable checks that a constant operand is representable in the given type.
func (check *checker) isRepresentable(x *operand, typ *Basic) {
	if x.mode != constant || isUntyped(typ) {
		return
	}

	if !isRepresentableConst(x.val, typ.Kind) {
		var msg string
		if isNumeric(x.typ) && isNumeric(typ) {
			msg = "%s overflows %s"
		} else {
			msg = "cannot convert %s to %s"
		}
		check.errorf(x.pos(), msg, x, typ)
		x.mode = invalid
	}
}

// convertUntyped attempts to set the type of an untyped value to the target type.
func (check *checker) convertUntyped(x *operand, target Type) {
	if x.mode == invalid || !isUntyped(x.typ) {
		return
	}

	// TODO(gri) Sloppy code - clean up. This function is central
	//           to assignment and expression checking.

	if isUntyped(target) {
		// both x and target are untyped
		xkind := x.typ.(*Basic).Kind
		tkind := target.(*Basic).Kind
		if isNumeric(x.typ) && isNumeric(target) {
			if xkind < tkind {
				x.typ = target
			}
		} else if xkind != tkind {
			goto Error
		}
		return
	}

	// typed target
	switch t := underlying(target).(type) {
	case *Basic:
		check.isRepresentable(x, t)
	case *Interface:
		if !x.isNil() && len(t.Methods) > 0 /* empty interfaces are ok */ {
			goto Error
		}
	case *Pointer, *Signature, *Slice, *Map, *Chan:
		if !x.isNil() {
			goto Error
		}
	}

	x.typ = target
	return

Error:
	check.errorf(x.pos(), "cannot convert %s to %s", x, target)
	x.mode = invalid
}

func (check *checker) comparison(x, y *operand, op token.Token) {
	// TODO(gri) deal with interface vs non-interface comparison

	valid := false
	if x.isAssignable(y.typ) || y.isAssignable(x.typ) {
		switch op {
		case token.EQL, token.NEQ:
			valid = isComparable(x.typ) ||
				x.isNil() && hasNil(y.typ) ||
				y.isNil() && hasNil(x.typ)
		case token.LSS, token.LEQ, token.GTR, token.GEQ:
			valid = isOrdered(x.typ)
		default:
			unreachable()
		}
	}

	if !valid {
		check.invalidOp(x.pos(), "cannot compare %s and %s", x, y)
		x.mode = invalid
		return
	}

	if x.mode == constant && y.mode == constant {
		x.val = compareConst(x.val, y.val, op)
	} else {
		x.mode = value
	}

	x.typ = Typ[UntypedBool]
}

// untyped lhs shift operands convert to the hint type
// TODO(gri) shift hinting is not correct
func (check *checker) shift(x, y *operand, op token.Token, hint Type) {
	// The right operand in a shift expression must have unsigned integer type
	// or be an untyped constant that can be converted to unsigned integer type.
	if y.mode == constant && isUntyped(y.typ) {
		if isRepresentableConst(y.val, UntypedInt) {
			y.typ = Typ[UntypedInt]
		}
	}
	if !isInteger(y.typ) || !isUnsigned(y.typ) && !isUntyped(y.typ) {
		check.invalidOp(y.pos(), "shift count %s must be unsigned integer", y)
		x.mode = invalid
		return
	}

	// If the left operand of a non-constant shift expression is an untyped
	// constant, the type of the constant is what it would be if the shift
	// expression were replaced by its left operand alone; the type is int
	// if it cannot be determined from the context (for instance, if the
	// shift expression is an operand in a comparison against an untyped
	// constant)
	if x.mode == constant && isUntyped(x.typ) {
		if y.mode == constant {
			// constant shift - accept values of any (untyped) type
			// as long as the value is representable as an integer
			if isRepresentableConst(x.val, UntypedInt) {
				x.typ = Typ[UntypedInt]
			}
		} else {
			// non-constant shift
			if hint != nil {
				check.convertUntyped(x, hint)
				if x.mode == invalid {
					return
				}
			}
		}
	}

	if !isInteger(x.typ) {
		check.invalidOp(x.pos(), "shifted operand %s must be integer", x)
		x.mode = invalid
		return
	}

	if y.mode == constant {
		const stupidShift = 1024
		s, ok := y.val.(int64)
		if !ok || s < 0 || s >= stupidShift {
			check.invalidOp(y.pos(), "%s: stupid shift", y)
			x.mode = invalid
			return
		}
		if x.mode == constant {
			x.val = shiftConst(x.val, uint(s), op)
			return
		}
		x.mode = value
	}

	// x.mode, x.Typ are unchanged
}

var binaryOpPredicates = opPredicates{
	token.ADD: func(typ Type) bool { return isNumeric(typ) || isString(typ) },
	token.SUB: isNumeric,
	token.MUL: isNumeric,
	token.QUO: isNumeric,
	token.REM: isInteger,

	token.AND:     isInteger,
	token.OR:      isInteger,
	token.XOR:     isInteger,
	token.AND_NOT: isInteger,

	token.LAND: isBoolean,
	token.LOR:  isBoolean,
}

func (check *checker) binary(x, y *operand, op token.Token, hint Type) {
	if isShift(op) {
		check.shift(x, y, op, hint)
		return
	}

	check.convertUntyped(x, y.typ)
	if x.mode == invalid {
		return
	}
	check.convertUntyped(y, x.typ)
	if y.mode == invalid {
		x.mode = invalid
		return
	}

	if isComparison(op) {
		check.comparison(x, y, op)
		return
	}

	if !isIdentical(x.typ, y.typ) {
		check.invalidOp(x.pos(), "mismatched types %s and %s", x.typ, y.typ)
		x.mode = invalid
		return
	}

	if !check.op(binaryOpPredicates, x, op) {
		x.mode = invalid
		return
	}

	if (op == token.QUO || op == token.REM) && y.mode == constant && isZeroConst(y.val) {
		check.invalidOp(y.pos(), "division by zero")
		x.mode = invalid
		return
	}

	if x.mode == constant && y.mode == constant {
		x.val = binaryOpConst(x.val, y.val, op, isInteger(x.typ))
		// Typed constants must be representable in
		// their type after each constant operation.
		check.isRepresentable(x, underlying(x.typ).(*Basic))
		return
	}

	x.mode = value
	// x.typ is unchanged
}

// index checks an index expression for validity. If length >= 0, it is the upper
// bound for the index. The result is a valid index >= 0, or a negative value.
//
func (check *checker) index(index ast.Expr, length int64, iota int) int64 {
	var x operand

	check.expr(&x, index, nil, iota)
	if !x.isInteger() {
		check.errorf(x.pos(), "index %s must be integer", &x)
		return -1
	}
	if x.mode != constant {
		return -1 // we cannot check more
	}
	// The spec doesn't require int64 indices, but perhaps it should.
	i, ok := x.val.(int64)
	if !ok {
		check.errorf(x.pos(), "stupid index %s", &x)
		return -1
	}
	if i < 0 {
		check.errorf(x.pos(), "index %s must not be negative", &x)
		return -1
	}
	if length >= 0 && i >= length {
		check.errorf(x.pos(), "index %s is out of bounds (>= %d)", &x, length)
		return -1
	}

	return i
}

// indexElts checks the elements (elts) of an array or slice composite literal
// against the literals element type (typ), and the element indices against
// the literal length if known (length >= 0). It returns the length of the
// literal (maximum index value + 1).
//
func (check *checker) indexedElts(elts []ast.Expr, typ Type, length int64, iota int) int64 {
	visited := make(map[int64]bool, len(elts))
	var index, max int64
	for _, e := range elts {
		// determine and check index
		validIndex := false
		eval := e
		if kv, _ := e.(*ast.KeyValueExpr); kv != nil {
			if i := check.index(kv.Key, length, iota); i >= 0 {
				index = i
				validIndex = true
			}
			eval = kv.Value
		} else if length >= 0 && index >= length {
			check.errorf(e.Pos(), "index %d is out of bounds (>= %d)", index, length)
		} else {
			validIndex = true
		}

		// if we have a valid index, check for duplicate entries
		if validIndex {
			if visited[index] {
				check.errorf(e.Pos(), "duplicate index %d in array or slice literal", index)
			}
			visited[index] = true
		}
		index++
		if index > max {
			max = index
		}

		// check element against composite literal element type
		var x operand
		check.expr(&x, eval, typ, iota)
		if !x.isAssignable(typ) {
			check.errorf(x.pos(), "cannot use %s as %s value in array or slice literal", &x, typ)
		}
	}
	return max
}

func (check *checker) argument(sig *Signature, i int, arg ast.Expr) {
	var par *ast.Object
	if n := len(sig.Params); i < n {
		par = sig.Params[i]
	} else if sig.IsVariadic {
		par = sig.Params[n-1]
	} else {
		check.errorf(arg.Pos(), "too many arguments")
		return
	}

	// TODO(gri) deal with ... last argument
	var z, x operand
	z.mode = variable
	z.expr = nil            // TODO(gri) can we do better here?
	z.typ = par.Type.(Type) // TODO(gri) should become something like checkObj(&z, ...) eventually
	check.expr(&x, arg, z.typ, -1)
	if x.mode == invalid {
		return // ignore this argument
	}
	check.assignOperand(&z, &x)
}

func (check *checker) recordType(x *operand) {
	if x.mode != invalid {
		check.mapf(x.expr, x.typ)
	}
}

// rawExpr typechecks expression e and initializes x with the expression
// value or type. If an error occured, x.mode is set to invalid.
// A hint != nil is used as operand type for untyped shifted operands;
// iota >= 0 indicates that the expression is part of a constant declaration.
// cycleOk indicates whether it is ok for a type expression to refer to itself.
//
func (check *checker) rawExpr(x *operand, e ast.Expr, hint Type, iota int, cycleOk bool) {
	if trace {
		check.trace(e.Pos(), "expr(%s, iota = %d, cycleOk = %v)", e, iota, cycleOk)
		defer check.untrace("=> %s", x)
	}

	if check.mapf != nil {
		defer check.recordType(x)
	}

	switch e := e.(type) {
	case *ast.BadExpr:
		goto Error // error was reported before

	case *ast.Ident:
		if e.Name == "_" {
			check.invalidOp(e.Pos(), "cannot use _ as value or type")
			goto Error
		}
		obj := e.Obj
		if obj == nil {
			goto Error // error was reported before
		}
		if obj.Type == nil {
			check.object(obj, cycleOk)
		}
		switch obj.Kind {
		case ast.Bad:
			goto Error // error was reported before
		case ast.Pkg:
			check.errorf(e.Pos(), "use of package %s not in selector", obj.Name)
			goto Error
		case ast.Con:
			if obj.Data == nil {
				goto Error // cycle detected
			}
			x.mode = constant
			if obj == universeIota {
				if iota < 0 {
					check.invalidAST(e.Pos(), "cannot use iota outside constant declaration")
					goto Error
				}
				x.val = int64(iota)
			} else {
				x.val = obj.Data
			}
		case ast.Typ:
			x.mode = typexpr
			if !cycleOk && underlying(obj.Type.(Type)) == nil {
				check.errorf(obj.Pos(), "illegal cycle in declaration of %s", obj.Name)
				x.expr = e
				x.typ = Typ[Invalid]
				return // don't goto Error - need x.mode == typexpr
			}
		case ast.Var:
			x.mode = variable
		case ast.Fun:
			x.mode = value
		default:
			unreachable()
		}
		x.typ = obj.Type.(Type)

	case *ast.Ellipsis:
		// ellipses are handled explictly where they are legal
		// (array composite literals and parameter lists)
		check.errorf(e.Pos(), "invalid use of '...'")
		goto Error

	case *ast.BasicLit:
		x.setConst(e.Kind, e.Value)
		if x.mode == invalid {
			check.invalidAST(e.Pos(), "invalid literal %v", e.Value)
			goto Error
		}

	case *ast.FuncLit:
		if typ, ok := check.typ(e.Type, false).(*Signature); ok {
			x.mode = value
			x.typ = typ
			check.function(typ, e.Body)
		} else {
			check.invalidAST(e.Pos(), "invalid function literal %s", e)
			goto Error
		}

	case *ast.CompositeLit:
		typ := hint
		openArray := false
		if e.Type != nil {
			// [...]T array types may only appear with composite literals.
			// Check for them here so we don't have to handle ... in general.
			typ = nil
			if atyp, _ := e.Type.(*ast.ArrayType); atyp != nil && atyp.Len != nil {
				if ellip, _ := atyp.Len.(*ast.Ellipsis); ellip != nil && ellip.Elt == nil {
					// We have an "open" [...]T array type.
					// Create a new ArrayType with unknown length (-1)
					// and finish setting it up after analyzing the literal.
					typ = &Array{Len: -1, Elt: check.typ(atyp.Elt, cycleOk)}
					openArray = true
				}
			}
			if typ == nil {
				typ = check.typ(e.Type, false)
			}
		}
		if typ == nil {
			check.errorf(e.Pos(), "missing type in composite literal")
			goto Error
		}

		switch utyp := underlying(deref(typ)).(type) {
		case *Struct:
			if len(e.Elts) == 0 {
				break
			}
			fields := utyp.Fields
			if _, ok := e.Elts[0].(*ast.KeyValueExpr); ok {
				// all elements must have keys
				visited := make([]bool, len(fields))
				for _, e := range e.Elts {
					kv, _ := e.(*ast.KeyValueExpr)
					if kv == nil {
						check.errorf(e.Pos(), "mixture of field:value and value elements in struct literal")
						continue
					}
					key, _ := kv.Key.(*ast.Ident)
					if key == nil {
						check.errorf(kv.Pos(), "invalid field name %s in struct literal", kv.Key)
						continue
					}
					i := utyp.fieldIndex(key.Name)
					if i < 0 {
						check.errorf(kv.Pos(), "unknown field %s in struct literal", key.Name)
						continue
					}
					// 0 <= i < len(fields)
					if visited[i] {
						check.errorf(kv.Pos(), "duplicate field name %s in struct literal", key.Name)
						continue
					}
					visited[i] = true
					check.expr(x, kv.Value, nil, iota)
					etyp := fields[i].Type
					if !x.isAssignable(etyp) {
						check.errorf(x.pos(), "cannot use %s as %s value in struct literal", x, etyp)
						continue
					}
				}
			} else {
				// no element must have a key
				for i, e := range e.Elts {
					if kv, _ := e.(*ast.KeyValueExpr); kv != nil {
						check.errorf(kv.Pos(), "mixture of field:value and value elements in struct literal")
						continue
					}
					check.expr(x, e, nil, iota)
					if i >= len(fields) {
						check.errorf(x.pos(), "too many values in struct literal")
						break // cannot continue
					}
					// i < len(fields)
					etyp := fields[i].Type
					if !x.isAssignable(etyp) {
						check.errorf(x.pos(), "cannot use %s as an element of type %s in struct literal", x, etyp)
						continue
					}
				}
				if len(e.Elts) < len(fields) {
					check.errorf(e.Rbrace, "too few values in struct literal")
					// ok to continue
				}
			}

		case *Array:
			n := check.indexedElts(e.Elts, utyp.Elt, utyp.Len, iota)
			// if we have an "open" [...]T array, set the length now that we know it
			if openArray {
				utyp.Len = n
			}

		case *Slice:
			check.indexedElts(e.Elts, utyp.Elt, -1, iota)

		case *Map:
			visited := make(map[interface{}]bool, len(e.Elts))
			for _, e := range e.Elts {
				kv, _ := e.(*ast.KeyValueExpr)
				if kv == nil {
					check.errorf(e.Pos(), "missing key in map literal")
					continue
				}
				check.expr(x, kv.Key, nil, iota)
				if !x.isAssignable(utyp.Key) {
					check.errorf(x.pos(), "cannot use %s as %s key in map literal", x, utyp.Key)
					continue
				}
				if x.mode == constant {
					if visited[x.val] {
						check.errorf(x.pos(), "duplicate key %s in map literal", x.val)
						continue
					}
					visited[x.val] = true
				}
				check.expr(x, kv.Value, utyp.Elt, iota)
				if !x.isAssignable(utyp.Elt) {
					check.errorf(x.pos(), "cannot use %s as %s value in map literal", x, utyp.Elt)
					continue
				}
			}

		default:
			check.errorf(e.Pos(), "%s is not a valid composite literal type", typ)
			goto Error
		}

		x.mode = variable // TODO(gri) mode is really a value - keep for now to get going
		x.typ = typ

	case *ast.ParenExpr:
		check.rawExpr(x, e.X, hint, iota, cycleOk)

	case *ast.SelectorExpr:
		sel := e.Sel.Name
		// If the identifier refers to a package, handle everything here
		// so we don't need a "package" mode for operands: package names
		// can only appear in qualified identifiers which are mapped to
		// selector expressions.
		if ident, ok := e.X.(*ast.Ident); ok {
			if obj := ident.Obj; obj != nil && obj.Kind == ast.Pkg {
				exp := obj.Data.(*ast.Scope).Lookup(sel)
				if exp == nil {
					check.errorf(e.Sel.Pos(), "cannot refer to unexported %s", sel)
					goto Error
				}
				// simplified version of the code for *ast.Idents:
				// imported objects are always fully initialized
				switch exp.Kind {
				case ast.Con:
					assert(exp.Data != nil)
					x.mode = constant
					x.val = exp.Data
				case ast.Typ:
					x.mode = typexpr
				case ast.Var:
					x.mode = variable
				case ast.Fun:
					x.mode = value
				default:
					unreachable()
				}
				x.expr = e
				x.typ = exp.Type.(Type)
				return
			}
		}

		check.exprOrType(x, e.X, nil, iota, false)
		if x.mode == invalid {
			goto Error
		}
		mode, typ := lookupField(x.typ, sel)
		if mode == invalid {
			check.invalidOp(e.Pos(), "%s has no single field or method %s", x, sel)
			goto Error
		}
		if x.mode == typexpr {
			// method expression
			sig, ok := typ.(*Signature)
			if !ok {
				check.invalidOp(e.Pos(), "%s has no method %s", x, sel)
				goto Error
			}
			// the receiver type becomes the type of the first function
			// argument of the method expression's function type
			// TODO(gri) at the moment, method sets don't correctly track
			// pointer vs non-pointer receivers => typechecker is too lenient
			arg := ast.NewObj(ast.Var, "")
			arg.Type = x.typ
			x.mode = value
			x.typ = &Signature{
				Params:     append(ObjList{arg}, sig.Params...),
				Results:    sig.Results,
				IsVariadic: sig.IsVariadic,
			}
		} else {
			// regular selector
			x.mode = mode
			x.typ = typ
		}

	case *ast.IndexExpr:
		check.expr(x, e.X, hint, iota)

		valid := false
		length := int64(-1) // valid if >= 0
		switch typ := underlying(x.typ).(type) {
		case *Basic:
			if isString(typ) {
				valid = true
				if x.mode == constant {
					length = int64(len(x.val.(string)))
				}
				// an indexed string always yields a byte value
				// (not a constant) even if the string and the
				// index are constant
				x.mode = value
				x.typ = Typ[Byte]
			}

		case *Array:
			valid = true
			length = typ.Len
			if x.mode != variable {
				x.mode = value
			}
			x.typ = typ.Elt

		case *Pointer:
			if typ, _ := underlying(typ.Base).(*Array); typ != nil {
				valid = true
				length = typ.Len
				x.mode = variable
				x.typ = typ.Elt
			}

		case *Slice:
			valid = true
			x.mode = variable
			x.typ = typ.Elt

		case *Map:
			var key operand
			check.expr(&key, e.Index, nil, iota)
			if key.mode == invalid || !key.isAssignable(typ.Key) {
				check.invalidOp(x.pos(), "cannot use %s as map index of type %s", &key, typ.Key)
				goto Error
			}
			x.mode = valueok
			x.typ = typ.Elt
			x.expr = e
			return
		}

		if !valid {
			check.invalidOp(x.pos(), "cannot index %s", x)
			goto Error
		}

		if e.Index == nil {
			check.invalidAST(e.Pos(), "missing index expression for %s", x)
			return
		}

		check.index(e.Index, length, iota)
		// ok to continue

	case *ast.SliceExpr:
		check.expr(x, e.X, hint, iota)

		valid := false
		length := int64(-1) // valid if >= 0
		switch typ := underlying(x.typ).(type) {
		case *Basic:
			if isString(typ) {
				valid = true
				if x.mode == constant {
					length = int64(len(x.val.(string))) + 1 // +1 for slice
				}
				// a sliced string always yields a string value
				// of the same type as the original string (not
				// a constant) even if the string and the indices
				// are constant
				x.mode = value
				// x.typ doesn't change
			}

		case *Array:
			valid = true
			length = typ.Len + 1 // +1 for slice
			if x.mode != variable {
				check.invalidOp(x.pos(), "cannot slice %s (value not addressable)", x)
				goto Error
			}
			x.typ = &Slice{Elt: typ.Elt}

		case *Pointer:
			if typ, _ := underlying(typ.Base).(*Array); typ != nil {
				valid = true
				length = typ.Len + 1 // +1 for slice
				x.mode = variable
				x.typ = &Slice{Elt: typ.Elt}
			}

		case *Slice:
			valid = true
			x.mode = variable
			// x.typ doesn't change
		}

		if !valid {
			check.invalidOp(x.pos(), "cannot slice %s", x)
			goto Error
		}

		lo := int64(0)
		if e.Low != nil {
			lo = check.index(e.Low, length, iota)
		}

		hi := int64(-1)
		if e.High != nil {
			hi = check.index(e.High, length, iota)
		} else if length >= 0 {
			hi = length
		}

		if lo >= 0 && hi >= 0 && lo > hi {
			check.errorf(e.Low.Pos(), "inverted slice range: %d > %d", lo, hi)
			// ok to continue
		}

	case *ast.TypeAssertExpr:
		check.expr(x, e.X, hint, iota)
		if x.mode == invalid {
			goto Error
		}
		var T *Interface
		if T, _ = underlying(x.typ).(*Interface); T == nil {
			check.invalidOp(x.pos(), "%s is not an interface", x)
			goto Error
		}
		// x.(type) expressions are handled explicitly in type switches
		if e.Type == nil {
			check.errorf(e.Pos(), "use of .(type) outside type switch")
			goto Error
		}
		typ := check.typ(e.Type, false)
		if typ == Typ[Invalid] {
			goto Error
		}
		if method, wrongType := missingMethod(typ, T); method != nil {
			var msg string
			if wrongType {
				msg = "%s cannot have dynamic type %s (wrong type for method %s)"
			} else {
				msg = "%s cannot have dynamic type %s (missing method %s)"
			}
			check.errorf(e.Type.Pos(), msg, x, typ, method.Name)
			// ok to continue
		}
		x.mode = valueok
		x.expr = e
		x.typ = typ

	case *ast.CallExpr:
		check.exprOrType(x, e.Fun, nil, iota, false)
		if x.mode == invalid {
			goto Error
		} else if x.mode == typexpr {
			check.conversion(x, e, x.typ, iota)
		} else if sig, ok := underlying(x.typ).(*Signature); ok {
			// check parameters
			// TODO(gri)
			// - deal with single multi-valued function arguments: f(g())
			// - variadic functions only partially addressed
			for i, arg := range e.Args {
				check.argument(sig, i, arg)
			}

			// determine result
			x.mode = value
			if len(sig.Results) == 1 {
				x.typ = sig.Results[0].Type.(Type)
			} else {
				// TODO(gri) change Signature representation to use tuples,
				//           then this conversion is not required
				list := make([]Type, len(sig.Results))
				for i, obj := range sig.Results {
					list[i] = obj.Type.(Type)
				}
				x.typ = &tuple{list: list}
			}

		} else if bin, ok := x.typ.(*builtin); ok {
			check.builtin(x, e, bin, iota)

		} else {
			check.invalidOp(x.pos(), "cannot call non-function %s", x)
			goto Error
		}

	case *ast.StarExpr:
		check.exprOrType(x, e.X, hint, iota, true)
		switch x.mode {
		case invalid:
			goto Error
		case typexpr:
			x.typ = &Pointer{Base: x.typ}
		default:
			if typ, ok := x.typ.(*Pointer); ok {
				x.mode = variable
				x.typ = typ.Base
			} else {
				check.invalidOp(x.pos(), "cannot indirect %s", x)
				goto Error
			}
		}

	case *ast.UnaryExpr:
		check.expr(x, e.X, hint, iota)
		check.unary(x, e.Op)

	case *ast.BinaryExpr:
		var y operand
		check.expr(x, e.X, hint, iota)
		check.expr(&y, e.Y, hint, iota)
		check.binary(x, &y, e.Op, hint)

	case *ast.KeyValueExpr:
		// key:value expressions are handled in composite literals
		check.invalidAST(e.Pos(), "no key:value expected")
		goto Error

	case *ast.ArrayType:
		if e.Len != nil {
			check.expr(x, e.Len, nil, iota)
			if x.mode == invalid {
				goto Error
			}
			if x.mode != constant {
				if x.mode != invalid {
					check.errorf(x.pos(), "array length %s must be constant", x)
				}
				goto Error
			}
			n, ok := x.val.(int64)
			if !ok || n < 0 {
				check.errorf(x.pos(), "invalid array length %s", x)
				goto Error
			}
			x.typ = &Array{Len: n, Elt: check.typ(e.Elt, cycleOk)}
		} else {
			x.typ = &Slice{Elt: check.typ(e.Elt, true)}
		}
		x.mode = typexpr

	case *ast.StructType:
		x.mode = typexpr
		x.typ = &Struct{Fields: check.collectFields(e.Fields, cycleOk)}

	case *ast.FuncType:
		params, isVariadic := check.collectParams(e.Params, true)
		results, _ := check.collectParams(e.Results, false)
		x.mode = typexpr
		x.typ = &Signature{Recv: nil, Params: params, Results: results, IsVariadic: isVariadic}

	case *ast.InterfaceType:
		x.mode = typexpr
		x.typ = &Interface{Methods: check.collectMethods(e.Methods)}

	case *ast.MapType:
		x.mode = typexpr
		x.typ = &Map{Key: check.typ(e.Key, true), Elt: check.typ(e.Value, true)}

	case *ast.ChanType:
		x.mode = typexpr
		x.typ = &Chan{Dir: e.Dir, Elt: check.typ(e.Value, true)}

	default:
		check.dump("e = %s", e)
		unreachable()
	}

	// everything went well
	x.expr = e
	return

Error:
	x.mode = invalid
	x.expr = e
}

// exprOrType is like rawExpr but reports an error if e doesn't represents a value or type.
func (check *checker) exprOrType(x *operand, e ast.Expr, hint Type, iota int, cycleOk bool) {
	check.rawExpr(x, e, hint, iota, cycleOk)
	if x.mode == novalue {
		check.errorf(x.pos(), "%s used as value or type", x)
		x.mode = invalid
	}
}

// expr is like rawExpr but reports an error if e doesn't represents a value.
func (check *checker) expr(x *operand, e ast.Expr, hint Type, iota int) {
	check.rawExpr(x, e, hint, iota, false)
	switch x.mode {
	case novalue:
		check.errorf(x.pos(), "%s used as value", x)
		x.mode = invalid
	case typexpr:
		check.errorf(x.pos(), "%s is not an expression", x)
		x.mode = invalid
	}
}

func (check *checker) rawTyp(e ast.Expr, cycleOk, nilOk bool) Type {
	var x operand
	check.rawExpr(&x, e, nil, -1, cycleOk)
	switch x.mode {
	case invalid:
		// ignore - error reported before
	case novalue:
		check.errorf(x.pos(), "%s used as type", &x)
	case typexpr:
		return x.typ
	case constant:
		if nilOk && x.isNil() {
			return nil
		}
		fallthrough
	default:
		check.errorf(x.pos(), "%s is not a type", &x)
	}
	return Typ[Invalid]
}

// typOrNil is like rawExpr but reports an error if e doesn't represents a type or the predeclared value nil.
// It returns e's type, nil, or Typ[Invalid] if an error occured.
//
func (check *checker) typOrNil(e ast.Expr, cycleOk bool) Type {
	return check.rawTyp(e, cycleOk, true)
}

// typ is like rawExpr but reports an error if e doesn't represents a type.
// It returns e's type, or Typ[Invalid] if an error occured.
//
func (check *checker) typ(e ast.Expr, cycleOk bool) Type {
	return check.rawTyp(e, cycleOk, false)
}
