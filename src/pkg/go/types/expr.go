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
// - use "" or "_" consistently for anonymous identifiers? (e.g. reeceivers that have no name)
// - consider storing error messages in invalid operands for better error messages/debugging output

// TODO(gri) API issues
// - clients need access to builtins type information
// - API tests are missing (e.g., identifiers should be handled as expressions in callbacks)

/*
Basic algorithm:

Expressions are checked recursively, top down. Expression checker functions
are generally of the form:

  func f(x *operand, e *ast.Expr, ...)

where e is the expression to be checked, and x is the result of the check.
The check performed by f may fail in which case x.mode == invalid, and
related error messages will have been issued by f.

If a hint argument is present, it is the composite literal element type
of an outer composite literal; it is used to type-check composite literal
elements that have no explicit type specification in the source
(e.g.: []T{{...}, {...}}, the hint is the type T in this case).

If an iota argument >= 0 is present, it is the value of iota for the
specific expression.

All expressions are checked via rawExpr, which dispatches according
to expression kind. Upon returning, rawExpr is recording the types and
constant values for all expressions that have an untyped type (those types
may change on the way up in the expression tree). Usually these are constants,
but the results of comparisons or non-constant shifts of untyped constants
may also be untyped, but not constant.

Untyped expressions may eventually become fully typed (i.e., not untyped),
typically when the value is assigned to a variable, or is used otherwise.
The updateExprType method is used to record this final type and update
the recorded types: the type-checked expression tree is again traversed down,
and the new type is propagated as needed. Untyped constant expression values
that become fully typed must now be representable by the full type (constant
sub-expression trees are left alone except for their roots). This mechanism
ensures that a client sees the actual (run-time) type an untyped value would
have. It also permits type-checking of lhs shift operands "as if the shift
were not present": when updateExprType visits an untyped lhs shift operand
and assigns it it's final type, that type must be an integer type, and a
constant lhs must be representable as an integer.

When an expression gets its final type, either on the way out from rawExpr,
on the way down in updateExprType, or at the end of the type checker run,
if present the Context.Expr method is invoked to notify a go/types client.
*/

func (check *checker) collectParams(list *ast.FieldList, variadicOk bool) (params []*Var, isVariadic bool) {
	if list == nil {
		return
	}
	var last *Var
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
				par := check.lookup(name).(*Var)
				par.Type = typ
				last = par
				copy := *par
				params = append(params, &copy)
			}
		} else {
			// anonymous parameter
			par := &Var{Type: typ}
			last = nil // not accessible inside function
			params = append(params, par)
		}
	}
	// For a variadic function, change the last parameter's object type
	// from T to []T (this is the type used inside the function), but
	// keep the params list unchanged (this is the externally visible type).
	if isVariadic && last != nil {
		last.Type = &Slice{Elt: last.Type}
	}
	return
}

func (check *checker) collectMethods(list *ast.FieldList) (methods []*Method) {
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
			sig, ok := typ.(*Signature)
			if !ok {
				check.invalidAST(f.Type.Pos(), "%s is not a method signature", typ)
				continue
			}
			for _, name := range f.Names {
				methods = append(methods, &Method{QualifiedName{check.pkg, name.Name}, sig})
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
	// Check for double declarations.
	// The parser inserts methods into an interface-local scope, so local
	// double declarations are reported by the parser already. We need to
	// check again for conflicts due to embedded interfaces. This will lead
	// to a 2nd error message if the double declaration was reported before
	// by the parser.
	// TODO(gri) clean this up a bit
	seen := make(map[string]bool)
	for _, m := range methods {
		if seen[m.Name] {
			check.errorf(list.Pos(), "multiple methods named %s", m.Name)
			return // keep multiple entries, lookup will only return the first entry
		}
		seen[m.Name] = true
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

func (check *checker) collectFields(list *ast.FieldList, cycleOk bool) (fields []*Field) {
	if list == nil {
		return
	}

	var typ Type   // current field typ
	var tag string // current field tag
	add := func(name string, isAnonymous bool) {
		fields = append(fields, &Field{QualifiedName{check.pkg, name}, typ, tag, isAnonymous})
	}

	for _, f := range list.List {
		typ = check.typ(f.Type, cycleOk)
		tag = check.tag(f.Tag)
		if len(f.Names) > 0 {
			// named fields
			for _, name := range f.Names {
				add(name.Name, false)
			}
		} else {
			// anonymous field
			switch t := deref(typ).(type) {
			case *Basic:
				add(t.Name, true)
			case *NamedType:
				add(t.Obj.GetName(), true)
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
		// spec: "As an exception to the addressability
		// requirement x may also be a composite literal."
		if _, ok := unparen(x.expr).(*ast.CompositeLit); ok {
			x.mode = variable
		}
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
		typ := underlying(x.typ).(*Basic)
		x.val = unaryOpConst(x.val, check.ctxt, op, typ)
		// Typed constants must be representable in
		// their type after each constant operation.
		check.isRepresentable(x, typ)
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

	if !isRepresentableConst(x.val, check.ctxt, typ.Kind) {
		var msg string
		if isNumeric(x.typ) && isNumeric(typ) {
			msg = "%s overflows (or cannot be accurately represented as) %s"
		} else {
			msg = "cannot convert %s to %s"
		}
		check.errorf(x.pos(), msg, x, typ)
		x.mode = invalid
	}
}

// updateExprType updates the type of x to typ and invokes itself
// recursively for the operands of x, depending on expression kind.
// If typ is still an untyped and not the final type, updateExprType
// only updates the recorded untyped type for x and possibly its
// operands. Otherwise (i.e., typ is not an untyped type anymore,
// or it is the final type for x), Context.Expr is invoked, if present.
// Also, if x is a constant, it must be representable as a value of typ,
// and if x is the (formerly untyped) lhs operand of a non-constant
// shift, it must be an integer value.
//
func (check *checker) updateExprType(x ast.Expr, typ Type, final bool) {
	old, found := check.untyped[x]
	if !found {
		return // nothing to do
	}

	// update operands of x if necessary
	switch x := x.(type) {
	case *ast.BadExpr,
		*ast.FuncLit,
		*ast.CompositeLit,
		*ast.IndexExpr,
		*ast.SliceExpr,
		*ast.TypeAssertExpr,
		*ast.StarExpr,
		*ast.KeyValueExpr,
		*ast.ArrayType,
		*ast.StructType,
		*ast.FuncType,
		*ast.InterfaceType,
		*ast.MapType,
		*ast.ChanType:
		// These expression are never untyped - nothing to do.
		// The respective sub-expressions got their final types
		// upon assignment or use.
		if debug {
			check.dump("%s: found old type(%s): %s (new: %s)", x.Pos(), x, old.typ, typ)
			unreachable()
		}
		return

	case *ast.CallExpr:
		// Resulting in an untyped constant (e.g., built-in complex).
		// The respective calls take care of calling updateExprType
		// for the arguments if necessary.

	case *ast.Ident, *ast.BasicLit, *ast.SelectorExpr:
		// An identifier denoting a constant, a constant literal,
		// or a qualified identifier (imported untyped constant).
		// No operands to take care of.

	case *ast.ParenExpr:
		check.updateExprType(x.X, typ, final)

	case *ast.UnaryExpr:
		// If x is a constant, the operands were constants.
		// They don't need to be updated since they never
		// get "materialized" into a typed value; and they
		// will be processed at the end of the type check.
		if old.isConst {
			break
		}
		check.updateExprType(x.X, typ, final)

	case *ast.BinaryExpr:
		if old.isConst {
			break // see comment for unary expressions
		}
		if isComparison(x.Op) {
			// The result type is independent of operand types
			// and the operand types must have final types.
		} else if isShift(x.Op) {
			// The result type depends only on lhs operand.
			// The rhs type was updated when checking the shift.
			check.updateExprType(x.X, typ, final)
		} else {
			// The operand types match the result type.
			check.updateExprType(x.X, typ, final)
			check.updateExprType(x.Y, typ, final)
		}

	default:
		unreachable()
	}

	// If the new type is not final and still untyped, just
	// update the recorded type.
	if !final && isUntyped(typ) {
		old.typ = underlying(typ).(*Basic)
		check.untyped[x] = old
		return
	}

	// Otherwise we have the final (typed or untyped type).
	// Remove it from the map.
	delete(check.untyped, x)

	// If x is the lhs of a shift, its final type must be integer.
	// We already know from the shift check that it is representable
	// as an integer if it is a constant.
	if old.isLhs && !isInteger(typ) {
		check.invalidOp(x.Pos(), "shifted operand %s (type %s) must be integer", x, typ)
		return
	}

	// Everything's fine, notify client of final type for x.
	if f := check.ctxt.Expr; f != nil {
		var val interface{}
		if old.isConst {
			val = old.val
		}
		f(x, typ, val)
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
				check.updateExprType(x.expr, target, false)
			}
		} else if xkind != tkind {
			goto Error
		}
		return
	}

	// typed target
	switch t := underlying(target).(type) {
	case nil:
		// We may reach here due to previous type errors.
		// Be conservative and don't crash.
		x.mode = invalid
		return
	case *Basic:
		check.isRepresentable(x, t)
		if x.mode == invalid {
			return // error already reported
		}
	case *Interface:
		if !x.isNil() && len(t.Methods) > 0 /* empty interfaces are ok */ {
			goto Error
		}
		// Update operand types to the default type rather then
		// the target (interface) type: values must have concrete
		// dynamic types. If the value is nil, keep it untyped
		// (this is important for tools such as go vet which need
		// the dynamic type for argument checking of say, print
		// functions)
		if x.isNil() {
			target = Typ[UntypedNil]
		} else {
			// cannot assign untyped values to non-empty interfaces
			if len(t.Methods) > 0 {
				goto Error
			}
			target = defaultType(x.typ)
		}
	case *Pointer, *Signature, *Slice, *Map, *Chan:
		if !x.isNil() {
			goto Error
		}
		// keep nil untyped - see comment for interfaces, above
		target = Typ[UntypedNil]
	default:
		if debug {
			check.dump("convertUntyped(x = %v, target = %v)", x, target)
		}
		unreachable()
	}

	x.typ = target
	check.updateExprType(x.expr, target, true) // UntypedNils are final
	return

Error:
	check.errorf(x.pos(), "cannot convert %s to %s", x, target)
	x.mode = invalid
}

func (check *checker) comparison(x, y *operand, op token.Token) {
	// TODO(gri) deal with interface vs non-interface comparison

	valid := false
	if x.isAssignable(check.ctxt, y.typ) || y.isAssignable(check.ctxt, x.typ) {
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
		check.invalidOp(x.pos(), "cannot compare %s %s %s", x, op, y)
		x.mode = invalid
		return
	}

	if x.mode == constant && y.mode == constant {
		x.val = compareConst(x.val, y.val, op)
	} else {
		x.mode = value
	}

	// The result type of a comparison is always boolean and
	// independent of the argument types. They have now their
	// final types (untyped or typed): update the respective
	// expression trees.
	check.updateExprType(x.expr, x.typ, true)
	check.updateExprType(y.expr, y.typ, true)

	x.typ = Typ[UntypedBool]
}

func (check *checker) shift(x, y *operand, op token.Token) {
	untypedx := isUntyped(x.typ)

	// The lhs must be of integer type or be representable
	// as an integer; otherwise the shift has no chance.
	if !isInteger(x.typ) && (!untypedx || !isRepresentableConst(x.val, nil, UntypedInt)) {
		check.invalidOp(x.pos(), "shifted operand %s must be integer", x)
		x.mode = invalid
		return
	}

	// spec: "The right operand in a shift expression must have unsigned
	// integer type or be an untyped constant that can be converted to
	// unsigned integer type."
	switch {
	case isInteger(y.typ) && isUnsigned(y.typ):
		// nothing to do
	case isUntyped(y.typ):
		check.convertUntyped(y, Typ[UntypedInt])
		if y.mode == invalid {
			x.mode = invalid
			return
		}
	default:
		check.invalidOp(y.pos(), "shift count %s must be unsigned integer", y)
		x.mode = invalid
		return
	}

	if x.mode == constant {
		if y.mode == constant {
			if untypedx {
				x.typ = Typ[UntypedInt]
			}
			if x.val != nil && y.val != nil {
				// rhs must be within reasonable bounds
				const stupidShift = 1024
				s, ok := y.val.(int64)
				if !ok || s < 0 || s >= stupidShift {
					check.invalidOp(y.pos(), "%s: stupid shift", y)
					x.mode = invalid
					return
				}
				// everything's ok
				x.val = shiftConst(x.val, uint(s), op)
			} else {
				x.val = nil
			}
			return
		}

		// non-constant shift with constant lhs
		if untypedx {
			// spec: "If the left operand of a non-constant shift expression is
			// an untyped constant, the type of the constant is what it would be
			// if the shift expression were replaced by its left operand alone;
			// the type is int if it cannot be determined from the context (for
			// instance, if the shift expression is an operand in a comparison
			// against an untyped constant)".

			// Delay operand checking until we know the final type:
			// The lhs expression must be in the untyped map, mark
			// the entry as lhs shift operand.
			if info, ok := check.untyped[x.expr]; ok {
				info.isLhs = true
				check.untyped[x.expr] = info
			} else {
				unreachable()
			}
			// keep x's type
			x.mode = value
			return
		}
	}

	// non-constant shift - lhs must be an integer
	if !isInteger(x.typ) {
		check.invalidOp(x.pos(), "shifted operand %s must be integer", x)
		x.mode = invalid
		return
	}

	// non-constant shift
	x.mode = value
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

func (check *checker) binary(x *operand, lhs, rhs ast.Expr, op token.Token, iota int) {
	var y operand

	check.expr(x, lhs, nil, iota)
	check.expr(&y, rhs, nil, iota)

	if x.mode == invalid {
		return
	}
	if y.mode == invalid {
		x.mode = invalid
		x.expr = y.expr
		return
	}

	if isShift(op) {
		check.shift(x, &y, op)
		return
	}

	check.convertUntyped(x, y.typ)
	if x.mode == invalid {
		return
	}
	check.convertUntyped(&y, x.typ)
	if y.mode == invalid {
		x.mode = invalid
		return
	}

	if isComparison(op) {
		check.comparison(x, &y, op)
		return
	}

	if !IsIdentical(x.typ, y.typ) {
		// only report an error if we have valid types
		// (otherwise we had an error reported elsewhere already)
		if x.typ != Typ[Invalid] && y.typ != Typ[Invalid] {
			check.invalidOp(x.pos(), "mismatched types %s and %s", x.typ, y.typ)
		}
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
		typ := underlying(x.typ).(*Basic)
		x.val = binaryOpConst(x.val, y.val, op, typ)
		// Typed constants must be representable in
		// their type after each constant operation.
		check.isRepresentable(x, typ)
		return
	}

	x.mode = value
	// x.typ is unchanged
}

// index checks an index/size expression arg for validity.
// If length >= 0, it is the upper bound for arg.
// TODO(gri): Do we need iota?
func (check *checker) index(arg ast.Expr, length int64, iota int) (i int64, ok bool) {
	var x operand
	check.expr(&x, arg, nil, iota)

	// an untyped constant must be representable as Int
	check.convertUntyped(&x, Typ[Int])
	if x.mode == invalid {
		return
	}

	// the index/size must be of integer type
	if !isInteger(x.typ) {
		check.invalidArg(x.pos(), "%s must be integer", &x)
		return
	}

	// a constant index/size i must be 0 <= i < length
	if x.mode == constant && x.val != nil {
		i = x.val.(int64)
		if i < 0 {
			check.invalidArg(x.pos(), "%s must not be negative", &x)
			return
		}
		if length >= 0 && i >= length {
			check.errorf(x.pos(), "index %s is out of bounds (>= %d)", &x, length)
			return
		}
		// 0 <= i [ && i < length ]
		return i, true
	}

	return -1, true
}

// compositeLitKey resolves unresolved composite literal keys.
// For details, see comment in go/parser/parser.go, method parseElement.
func (check *checker) compositeLitKey(key ast.Expr) {
	if ident, ok := key.(*ast.Ident); ok && ident.Obj == nil {
		if obj := check.pkg.Scope.Lookup(ident.Name); obj != nil {
			check.register(ident, obj)
		} else if obj := Universe.Lookup(ident.Name); obj != nil {
			check.register(ident, obj)
		} else {
			check.errorf(ident.Pos(), "undeclared name: %s", ident.Name)
		}
	}
}

// indexElts checks the elements (elts) of an array or slice composite literal
// against the literal's element type (typ), and the element indices against
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
			check.compositeLitKey(kv.Key)
			if i, ok := check.index(kv.Key, length, iota); ok {
				if i >= 0 {
					index = i
				}
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
		if !check.assignment(&x, typ) && x.mode != invalid {
			check.errorf(x.pos(), "cannot use %s as %s value in array or slice literal", &x, typ)
		}
	}
	return max
}

// argument typechecks passing an argument arg (if arg != nil) or
// x (if arg == nil) to the i'th parameter of the given signature.
// If passSlice is set, the argument is followed by ... in the call.
//
func (check *checker) argument(sig *Signature, i int, arg ast.Expr, x *operand, passSlice bool) {
	// determine parameter
	var par *Var
	n := len(sig.Params)
	if i < n {
		par = sig.Params[i]
	} else if sig.IsVariadic {
		par = sig.Params[n-1]
	} else {
		check.errorf(arg.Pos(), "too many arguments")
		return
	}

	// determine argument
	var z operand
	z.mode = variable
	z.expr = nil // TODO(gri) can we do better here? (for good error messages)
	z.typ = par.Type

	if arg != nil {
		check.expr(x, arg, z.typ, -1)
	}
	if x.mode == invalid {
		return // ignore this argument
	}

	// check last argument of the form x...
	if passSlice {
		if i+1 != n {
			check.errorf(x.pos(), "can only use ... with matching parameter")
			return // ignore this argument
		}
		// spec: "If the final argument is assignable to a slice type []T,
		// it may be passed unchanged as the value for a ...T parameter if
		// the argument is followed by ..."
		z.typ = &Slice{Elt: z.typ} // change final parameter type to []T
	}

	if !check.assignment(x, z.typ) && x.mode != invalid {
		check.errorf(x.pos(), "cannot pass argument %s to %s", x, &z)
	}
}

var emptyResult Result

func (check *checker) callExpr(x *operand) {
	// convert x into a user-friendly set of values
	var typ Type
	var val interface{}
	switch x.mode {
	case invalid:
		return // nothing to do
	case novalue:
		typ = &emptyResult
	case constant:
		typ = x.typ
		val = x.val
	default:
		typ = x.typ
	}

	// if the operand is untyped, delay notification
	// until it becomes typed or until the end of
	// type checking
	if isUntyped(typ) {
		check.untyped[x.expr] = exprInfo{x.mode == constant, false, typ.(*Basic), val}
		return
	}

	// TODO(gri) ensure that literals always report
	// their dynamic (never interface) type.
	// This is not the case yet.

	if check.ctxt.Expr != nil {
		check.ctxt.Expr(x.expr, typ, val)
	}
}

// rawExpr typechecks expression e and initializes x with the expression
// value or type. If an error occurred, x.mode is set to invalid.
// If hint != nil, it is the type of a composite literal element.
// iota >= 0 indicates that the expression is part of a constant declaration.
// cycleOk indicates whether it is ok for a type expression to refer to itself.
//
func (check *checker) rawExpr(x *operand, e ast.Expr, hint Type, iota int, cycleOk bool) {
	if trace {
		c := ""
		if cycleOk {
			c = " ⨁"
		}
		check.trace(e.Pos(), "%s (%s, %d%s)", e, typeString(hint), iota, c)
		defer check.untrace("=> %s", x)
	}

	// record final type of x if untyped, notify clients of type otherwise
	defer check.callExpr(x)

	switch e := e.(type) {
	case *ast.BadExpr:
		goto Error // error was reported before

	case *ast.Ident:
		if e.Name == "_" {
			check.invalidOp(e.Pos(), "cannot use _ as value or type")
			goto Error
		}
		obj := check.lookup(e)
		if obj == nil {
			goto Error // error was reported before
		}
		check.object(obj, cycleOk)
		switch obj := obj.(type) {
		case *Package:
			check.errorf(e.Pos(), "use of package %s not in selector", obj.Name)
			goto Error
		case *Const:
			if obj.Type == Typ[Invalid] {
				goto Error
			}
			x.mode = constant
			if obj == universeIota {
				if iota < 0 {
					check.invalidAST(e.Pos(), "cannot use iota outside constant declaration")
					goto Error
				}
				x.val = int64(iota)
			} else {
				x.val = obj.Val // may be nil if we don't know the constant value
			}
		case *TypeName:
			x.mode = typexpr
			if !cycleOk && underlying(obj.Type) == nil {
				check.errorf(obj.spec.Pos(), "illegal cycle in declaration of %s", obj.Name)
				x.expr = e
				x.typ = Typ[Invalid]
				return // don't goto Error - need x.mode == typexpr
			}
		case *Var:
			x.mode = variable
		case *Func:
			x.mode = value
		default:
			unreachable()
		}
		x.typ = obj.GetType()

	case *ast.Ellipsis:
		// ellipses are handled explicitly where they are legal
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
		if sig, ok := check.typ(e.Type, false).(*Signature); ok {
			x.mode = value
			x.typ = sig
			check.later(nil, sig, e.Body)
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
					i := utyp.fieldIndex(QualifiedName{check.pkg, key.Name})
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
					if !check.assignment(x, etyp) {
						if x.mode != invalid {
							check.errorf(x.pos(), "cannot use %s as %s value in struct literal", x, etyp)
						}
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
					if !check.assignment(x, etyp) {
						if x.mode != invalid {
							check.errorf(x.pos(), "cannot use %s as %s value in struct literal", x, etyp)
						}
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
				check.compositeLitKey(kv.Key)
				check.expr(x, kv.Key, nil, iota)
				if !check.assignment(x, utyp.Key) {
					if x.mode != invalid {
						check.errorf(x.pos(), "cannot use %s as %s key in map literal", x, utyp.Key)
					}
					continue
				}
				if x.mode == constant && x.val != nil {
					if visited[x.val] {
						check.errorf(x.pos(), "duplicate key %s in map literal", x.val)
						continue
					}
					visited[x.val] = true
				}
				check.expr(x, kv.Value, utyp.Elt, iota)
				if !check.assignment(x, utyp.Elt) {
					if x.mode != invalid {
						check.errorf(x.pos(), "cannot use %s as %s value in map literal", x, utyp.Elt)
					}
					continue
				}
			}

		default:
			check.errorf(e.Pos(), "%s is not a valid composite literal type", typ)
			goto Error
		}

		x.mode = value
		x.typ = typ

	case *ast.ParenExpr:
		check.rawExpr(x, e.X, nil, iota, cycleOk)

	case *ast.SelectorExpr:
		sel := e.Sel.Name
		// If the identifier refers to a package, handle everything here
		// so we don't need a "package" mode for operands: package names
		// can only appear in qualified identifiers which are mapped to
		// selector expressions.
		if ident, ok := e.X.(*ast.Ident); ok {
			if pkg, ok := check.lookup(ident).(*Package); ok {
				exp := pkg.Scope.Lookup(sel)
				// gcimported package scopes contain non-exported
				// objects such as types used in partially exported
				// objects - do not accept them
				if exp == nil || !ast.IsExported(exp.GetName()) {
					check.errorf(e.Pos(), "cannot refer to unexported %s", e)
					goto Error
				}
				check.register(e.Sel, exp)
				// Simplified version of the code for *ast.Idents:
				// - imported packages use types.Scope and types.Objects
				// - imported objects are always fully initialized
				switch exp := exp.(type) {
				case *Const:
					assert(exp.Val != nil)
					x.mode = constant
					x.typ = exp.Type
					x.val = exp.Val
				case *TypeName:
					x.mode = typexpr
					x.typ = exp.Type
				case *Var:
					x.mode = variable
					x.typ = exp.Type
				case *Func:
					x.mode = value
					x.typ = exp.Type
				default:
					unreachable()
				}
				x.expr = e
				return
			}
		}

		check.exprOrType(x, e.X, iota, false)
		if x.mode == invalid {
			goto Error
		}
		res := lookupField(x.typ, QualifiedName{check.pkg, sel})
		if res.mode == invalid {
			check.invalidOp(e.Pos(), "%s has no single field or method %s", x, sel)
			goto Error
		}
		if x.mode == typexpr {
			// method expression
			sig, ok := res.typ.(*Signature)
			if !ok {
				check.invalidOp(e.Pos(), "%s has no method %s", x, sel)
				goto Error
			}
			// the receiver type becomes the type of the first function
			// argument of the method expression's function type
			// TODO(gri) at the moment, method sets don't correctly track
			// pointer vs non-pointer receivers => typechecker is too lenient
			x.mode = value
			x.typ = &Signature{
				Params:     append([]*Var{{Type: x.typ}}, sig.Params...),
				Results:    sig.Results,
				IsVariadic: sig.IsVariadic,
			}
		} else {
			// regular selector
			x.mode = res.mode
			x.typ = res.typ
		}

	case *ast.IndexExpr:
		check.expr(x, e.X, nil, iota)
		if x.mode == invalid {
			goto Error
		}

		valid := false
		length := int64(-1) // valid if >= 0
		switch typ := underlying(x.typ).(type) {
		case *Basic:
			if isString(typ) {
				valid = true
				if x.mode == constant && x.val != nil {
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
			if !check.assignment(&key, typ.Key) {
				if key.mode != invalid {
					check.invalidOp(key.pos(), "cannot use %s as map index of type %s", &key, typ.Key)
				}
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
		check.expr(x, e.X, nil, iota)
		if x.mode == invalid {
			goto Error
		}

		valid := false
		length := int64(-1) // valid if >= 0
		switch typ := underlying(x.typ).(type) {
		case *Basic:
			if isString(typ) {
				valid = true
				if x.mode == constant && x.val != nil {
					length = int64(len(x.val.(string))) + 1 // +1 for slice
				}
				// a sliced string always yields a string value
				// of the same type as the original string (not
				// a constant) even if the string and the indices
				// are constant
				x.mode = value
				// x.typ doesn't change, but if it is an untyped
				// string it becomes string (see also issue 4913).
				if typ.Kind == UntypedString {
					x.typ = Typ[String]
				}
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
			if i, ok := check.index(e.Low, length, iota); ok && i >= 0 {
				lo = i
			}
		}

		hi := int64(-1)
		if e.High != nil {
			if i, ok := check.index(e.High, length, iota); ok && i >= 0 {
				hi = i
			}
		} else if length >= 0 {
			hi = length
		}

		if lo >= 0 && hi >= 0 && lo > hi {
			check.errorf(e.Low.Pos(), "inverted slice range: %d > %d", lo, hi)
			// ok to continue
		}

	case *ast.TypeAssertExpr:
		check.expr(x, e.X, nil, iota)
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
		check.exprOrType(x, e.Fun, iota, false)
		if x.mode == invalid {
			goto Error
		} else if x.mode == typexpr {
			check.conversion(x, e, x.typ, iota)
		} else if sig, ok := underlying(x.typ).(*Signature); ok {
			// check parameters

			// If we have a trailing ... at the end of the parameter
			// list, the last argument must match the parameter type
			// []T of a variadic function parameter x ...T.
			passSlice := false
			if e.Ellipsis.IsValid() {
				if sig.IsVariadic {
					passSlice = true
				} else {
					check.errorf(e.Ellipsis, "cannot use ... in call to %s", e.Fun)
					// ok to continue
				}
			}

			// If we have a single argument that is a function call
			// we need to handle it separately. Determine if this
			// is the case without checking the argument.
			var call *ast.CallExpr
			if len(e.Args) == 1 {
				call, _ = unparen(e.Args[0]).(*ast.CallExpr)
			}

			n := 0 // parameter count
			if call != nil {
				// We have a single argument that is a function call.
				check.expr(x, call, nil, -1)
				if x.mode == invalid {
					goto Error // TODO(gri): we can do better
				}
				if t, _ := x.typ.(*Result); t != nil {
					// multiple result values
					n = len(t.Values)
					for i, obj := range t.Values {
						x.mode = value
						x.expr = nil // TODO(gri) can we do better here? (for good error messages)
						x.typ = obj.Type
						check.argument(sig, i, nil, x, passSlice && i+1 == n)
					}
				} else {
					// single result value
					n = 1
					check.argument(sig, 0, nil, x, passSlice)
				}

			} else {
				// We don't have a single argument or it is not a function call.
				n = len(e.Args)
				for i, arg := range e.Args {
					check.argument(sig, i, arg, x, passSlice && i+1 == n)
				}
			}

			// determine if we have enough arguments
			if sig.IsVariadic {
				// a variadic function accepts an "empty"
				// last argument: count one extra
				n++
			}
			if n < len(sig.Params) {
				check.errorf(e.Fun.Pos(), "too few arguments in call to %s", e.Fun)
				// ok to continue
			}

			// determine result
			switch len(sig.Results) {
			case 0:
				x.mode = novalue
			case 1:
				x.mode = value
				x.typ = sig.Results[0].Type
			default:
				x.mode = value
				x.typ = &Result{Values: sig.Results}
			}

		} else if bin, ok := x.typ.(*builtin); ok {
			check.builtin(x, e, bin, iota)

		} else {
			check.invalidOp(x.pos(), "cannot call non-function %s", x)
			goto Error
		}

	case *ast.StarExpr:
		check.exprOrType(x, e.X, iota, true)
		switch x.mode {
		case invalid:
			goto Error
		case typexpr:
			x.typ = &Pointer{Base: x.typ}
		default:
			if typ, ok := underlying(x.typ).(*Pointer); ok {
				x.mode = variable
				x.typ = typ.Base
			} else {
				check.invalidOp(x.pos(), "cannot indirect %s", x)
				goto Error
			}
		}

	case *ast.UnaryExpr:
		check.expr(x, e.X, nil, iota)
		if x.mode == invalid {
			goto Error
		}
		check.unary(x, e.Op)
		if x.mode == invalid {
			goto Error
		}

	case *ast.BinaryExpr:
		check.binary(x, e.X, e.Y, e.Op, iota)
		if x.mode == invalid {
			goto Error
		}

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
		if debug {
			check.dump("expr = %v (%T)", e, e)
		}
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
func (check *checker) exprOrType(x *operand, e ast.Expr, iota int, cycleOk bool) {
	check.rawExpr(x, e, nil, iota, cycleOk)
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
// It returns e's type, nil, or Typ[Invalid] if an error occurred.
//
func (check *checker) typOrNil(e ast.Expr, cycleOk bool) Type {
	return check.rawTyp(e, cycleOk, true)
}

// typ is like rawExpr but reports an error if e doesn't represents a type.
// It returns e's type, or Typ[Invalid] if an error occurred.
//
func (check *checker) typ(e ast.Expr, cycleOk bool) Type {
	return check.rawTyp(e, cycleOk, false)
}
