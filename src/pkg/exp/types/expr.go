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

// TODO(gri)
// - don't print error messages referring to invalid types (they are likely spurious errors)
// - simplify invalid handling: maybe just use Typ[Invalid] as marker, get rid of invalid Mode for values?
// - rethink error handling: should all callers check if x.mode == valid after making a call?

func (check *checker) tag(field *ast.Field) string {
	if t := field.Tag; t != nil {
		assert(t.Kind == token.STRING)
		if tag, err := strconv.Unquote(t.Value); err == nil {
			return tag
		}
		check.invalidAST(t.Pos(), "incorrect tag syntax: %q", t.Value)
	}
	return ""
}

// collectFields collects interface methods (tok = token.INTERFACE), and function arguments/results (tok = token.FUNC).
func (check *checker) collectFields(tok token.Token, list *ast.FieldList, cycleOk bool) (fields ObjList, tags []string, isVariadic bool) {
	if list != nil {
		for _, field := range list.List {
			ftype := field.Type
			if t, ok := ftype.(*ast.Ellipsis); ok {
				ftype = t.Elt
				isVariadic = true
			}
			typ := check.typ(ftype, cycleOk)
			tag := check.tag(field)
			if len(field.Names) > 0 {
				// named fields
				for _, name := range field.Names {
					obj := name.Obj
					obj.Type = typ
					fields = append(fields, obj)
					if tok == token.STRUCT {
						tags = append(tags, tag)
					}
				}
			} else {
				// anonymous field
				switch tok {
				case token.FUNC:
					obj := ast.NewObj(ast.Var, "")
					obj.Type = typ
					fields = append(fields, obj)
				case token.INTERFACE:
					utyp := underlying(typ)
					if typ, ok := utyp.(*Interface); ok {
						// TODO(gri) This is not good enough. Check for double declarations!
						fields = append(fields, typ.Methods...)
					} else if utyp != Typ[Invalid] {
						// if utyp is invalid, don't complain (the root cause was reported before)
						check.errorf(ftype.Pos(), "interface contains embedded non-interface type")
					}
				default:
					panic("unreachable")
				}
			}
		}
	}
	return
}

func (check *checker) collectStructFields(list *ast.FieldList, cycleOk bool) (fields []*StructField) {
	if list == nil {
		return
	}
	for _, f := range list.List {
		typ := check.typ(f.Type, cycleOk)
		tag := check.tag(f)
		if len(f.Names) > 0 {
			// named fields
			for _, name := range f.Names {
				fields = append(fields, &StructField{name.Name, typ, tag, false})
			}
		} else {
			// anonymous field
			switch t := deref(typ).(type) {
			case *Basic:
				fields = append(fields, &StructField{t.Name, t, tag, true})
			case *NamedType:
				fields = append(fields, &StructField{t.Obj.Name, t, tag, true})
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
			// TODO(gri) better error message for <-x where x is a send-only channel
			//           (<- is defined but not permitted). Special-case here or
			//           handle higher up.
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
		check.isRepresentable(x, x.typ.(*Basic))
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
// bound for the index. The result is a valid integer constant, or nil.
//
func (check *checker) index(index ast.Expr, length int64, iota int) interface{} {
	var x operand

	check.expr(&x, index, nil, iota)
	if !x.isInteger() {
		check.errorf(x.pos(), "index %s must be integer", &x)
		return nil
	}
	if x.mode != constant {
		return nil // we cannot check more
	}
	// x.mode == constant and the index value must be >= 0
	if isNegConst(x.val) {
		check.errorf(x.pos(), "index %s must not be negative", &x)
		return nil
	}
	// x.val >= 0
	if length >= 0 && compareConst(x.val, length, token.GEQ) {
		check.errorf(x.pos(), "index %s is out of bounds (>= %d)", &x, length)
		return nil
	}

	return x.val
}

func (check *checker) callRecord(x *operand) {
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
		defer check.callRecord(x)
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
		unimplemented()

	case *ast.BasicLit:
		x.setConst(e.Kind, e.Value)
		if x.mode == invalid {
			check.invalidAST(e.Pos(), "invalid literal %v", e.Value)
			goto Error
		}

	case *ast.FuncLit:
		x.mode = value
		x.typ = check.typ(e.Type, false)
		// TODO(gri) handle errors (e.g. x.typ is not a *Signature)
		check.function(x.typ.(*Signature), e.Body)

	case *ast.CompositeLit:
		// TODO(gri)
		//	- determine element type if nil
		//	- deal with map elements
		var typ Type
		if e.Type != nil {
			// TODO(gri) Fix this - just to get going for now
			typ = check.typ(e.Type, false)
		}
		for _, e := range e.Elts {
			var x operand
			check.expr(&x, e, hint, iota)
			// TODO(gri) check assignment compatibility to element type
		}
		// TODO(gri) this is not correct - leave for now to get going
		x.mode = variable
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
			check.invalidOp(e.Pos(), "%s has no field or method %s", x, sel)
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
			// pointer vs non-pointer receivers -> typechecker is too lenient
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

		case *Slice:
			valid = true
			x.mode = variable
			x.typ = typ.Elt

		case *Map:
			// TODO(gri) check index type
			x.mode = valueok
			x.typ = typ.Elt
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
				// a constant) even if the string and the indexes
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

		case *Slice:
			valid = true
			x.mode = variable
			// x.typ doesn't change
		}

		if !valid {
			check.invalidOp(x.pos(), "cannot slice %s", x)
			goto Error
		}

		var lo interface{} = zeroConst
		if e.Low != nil {
			lo = check.index(e.Low, length, iota)
		}

		var hi interface{}
		if e.High != nil {
			hi = check.index(e.High, length, iota)
		} else if length >= 0 {
			hi = length
		}

		if lo != nil && hi != nil && compareConst(lo, hi, token.GTR) {
			check.errorf(e.Low.Pos(), "inverted slice range: %v > %v", lo, hi)
			// ok to continue
		}

	case *ast.TypeAssertExpr:
		check.expr(x, e.X, hint, iota)
		if _, ok := underlying(x.typ).(*Interface); !ok {
			check.invalidOp(e.X.Pos(), "non-interface type %s in type assertion", x.typ)
			// ok to continue
		}
		// TODO(gri) some type asserts are compile-time decidable
		x.mode = valueok
		x.expr = e
		x.typ = check.typ(e.Type, false)

	case *ast.CallExpr:
		check.exprOrType(x, e.Fun, nil, iota, false)
		if x.mode == invalid {
			goto Error
		} else if x.mode == typexpr {
			check.conversion(x, e, x.typ, iota)
		} else if sig, ok := underlying(x.typ).(*Signature); ok {
			// check parameters
			// TODO(gri) complete this
			// - deal with various forms of calls
			// - handle variadic calls
			if len(sig.Params) == len(e.Args) {
				var z, x operand
				z.mode = variable
				for i, arg := range e.Args {
					z.expr = nil                      // TODO(gri) can we do better here?
					z.typ = sig.Params[i].Type.(Type) // TODO(gri) should become something like checkObj(&z, ...) eventually
					check.expr(&x, arg, z.typ, iota)
					if x.mode == invalid {
						goto Error
					}
					check.assignOperand(&z, &x)
				}
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
		unimplemented()

	case *ast.ArrayType:
		if e.Len != nil {
			var n int64 = -1
			if ellip, ok := e.Len.(*ast.Ellipsis); ok {
				// TODO(gri) need to check somewhere that [...]T types are only used with composite literals
				if ellip.Elt != nil {
					check.invalidAST(ellip.Pos(), "ellipsis only expected")
					// ok to continue
				}
			} else {
				check.expr(x, e.Len, nil, 0)
				if x.mode == invalid {
					goto Error
				}
				if x.mode == constant {
					if i, ok := x.val.(int64); ok && i == int64(int(i)) {
						n = i
					}
				}
				if n < 0 {
					check.errorf(e.Len.Pos(), "invalid array bound %s", e.Len)
					// ok to continue
					n = 0
				}
			}
			x.typ = &Array{Len: n, Elt: check.typ(e.Elt, cycleOk)}
		} else {
			x.typ = &Slice{Elt: check.typ(e.Elt, true)}
		}
		x.mode = typexpr

	case *ast.StructType:
		x.mode = typexpr
		x.typ = &Struct{Fields: check.collectStructFields(e.Fields, cycleOk)}

	case *ast.FuncType:
		params, _, isVariadic := check.collectFields(token.FUNC, e.Params, true)
		results, _, _ := check.collectFields(token.FUNC, e.Results, true)
		x.mode = typexpr
		x.typ = &Signature{Recv: nil, Params: params, Results: results, IsVariadic: isVariadic}

	case *ast.InterfaceType:
		methods, _, _ := check.collectFields(token.INTERFACE, e.Methods, cycleOk)
		methods.Sort()
		x.mode = typexpr
		x.typ = &Interface{Methods: methods}

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

// expr is like rawExpr but reports an error if e doesn't represents a type.
// It returns e's type, or Typ[Invalid] if an error occured.
//
func (check *checker) typ(e ast.Expr, cycleOk bool) Type {
	var x operand
	check.rawExpr(&x, e, nil, -1, cycleOk)
	switch x.mode {
	case invalid:
		// ignore - error reported before
	case novalue:
		check.errorf(x.pos(), "%s used as type", &x)
	case typexpr:
		return x.typ
	default:
		check.errorf(x.pos(), "%s is not a type", &x)
	}
	return Typ[Invalid]
}
