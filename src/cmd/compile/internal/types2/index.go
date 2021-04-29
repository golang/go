// Copyright 2021 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file implements typechecking of index/slice expressions.

package types2

import (
	"cmd/compile/internal/syntax"
	"go/constant"
)

// If e is a valid function instantiation, indexExpr returns true.
// In that case x represents the uninstantiated function value and
// it is the caller's responsibility to instantiate the function.
func (check *Checker) indexExpr(x *operand, e *syntax.IndexExpr) (isFuncInst bool) {
	check.exprOrType(x, e.X)

	switch x.mode {
	case invalid:
		check.use(e.Index)
		return false

	case typexpr:
		// type instantiation
		x.mode = invalid
		x.typ = check.varType(e)
		if x.typ != Typ[Invalid] {
			x.mode = typexpr
		}
		return false

	case value:
		if sig := asSignature(x.typ); sig != nil && len(sig.tparams) > 0 {
			// function instantiation
			return true
		}
	}

	// ordinary index expression
	valid := false
	length := int64(-1) // valid if >= 0
	switch typ := optype(x.typ).(type) {
	case *Basic:
		if isString(typ) {
			valid = true
			if x.mode == constant_ {
				length = int64(len(constant.StringVal(x.val)))
			}
			// an indexed string always yields a byte value
			// (not a constant) even if the string and the
			// index are constant
			x.mode = value
			x.typ = universeByte // use 'byte' name
		}

	case *Array:
		valid = true
		length = typ.len
		if x.mode != variable {
			x.mode = value
		}
		x.typ = typ.elem

	case *Pointer:
		if typ := asArray(typ.base); typ != nil {
			valid = true
			length = typ.len
			x.mode = variable
			x.typ = typ.elem
		}

	case *Slice:
		valid = true
		x.mode = variable
		x.typ = typ.elem

	case *Map:
		index := check.singleIndex(e)
		if index == nil {
			x.mode = invalid
			return
		}
		var key operand
		check.expr(&key, index)
		check.assignment(&key, typ.key, "map index")
		// ok to continue even if indexing failed - map element type is known
		x.mode = mapindex
		x.typ = typ.elem
		x.expr = e
		return

	case *Sum:
		// A sum type can be indexed if all of the sum's types
		// support indexing and have the same index and element
		// type. Special rules apply for maps in the sum type.
		var tkey, telem Type // key is for map types only
		nmaps := 0           // number of map types in sum type
		if typ.is(func(t Type) bool {
			var e Type
			switch t := under(t).(type) {
			case *Basic:
				if isString(t) {
					e = universeByte
				}
			case *Array:
				e = t.elem
			case *Pointer:
				if t := asArray(t.base); t != nil {
					e = t.elem
				}
			case *Slice:
				e = t.elem
			case *Map:
				// If there are multiple maps in the sum type,
				// they must have identical key types.
				// TODO(gri) We may be able to relax this rule
				// but it becomes complicated very quickly.
				if tkey != nil && !Identical(t.key, tkey) {
					return false
				}
				tkey = t.key
				e = t.elem
				nmaps++
			case *TypeParam:
				check.errorf(x, "type of %s contains a type parameter - cannot index (implementation restriction)", x)
			case *instance:
				panic("unimplemented")
			}
			if e == nil || telem != nil && !Identical(e, telem) {
				return false
			}
			telem = e
			return true
		}) {
			// If there are maps, the index expression must be assignable
			// to the map key type (as for simple map index expressions).
			if nmaps > 0 {
				index := check.singleIndex(e)
				if index == nil {
					x.mode = invalid
					return
				}
				var key operand
				check.expr(&key, index)
				check.assignment(&key, tkey, "map index")
				// ok to continue even if indexing failed - map element type is known

				// If there are only maps, we are done.
				if nmaps == len(typ.types) {
					x.mode = mapindex
					x.typ = telem
					x.expr = e
					return
				}

				// Otherwise we have mix of maps and other types. For
				// now we require that the map key be an integer type.
				// TODO(gri) This is probably not good enough.
				valid = isInteger(tkey)
				// avoid 2nd indexing error if indexing failed above
				if !valid && key.mode == invalid {
					x.mode = invalid
					return
				}
				x.mode = value // map index expressions are not addressable
			} else {
				// no maps
				valid = true
				x.mode = variable
			}
			x.typ = telem
		}
	}

	if !valid {
		check.errorf(x, invalidOp+"cannot index %s", x)
		x.mode = invalid
		return
	}

	index := check.singleIndex(e)
	if index == nil {
		x.mode = invalid
		return
	}

	// In pathological (invalid) cases (e.g.: type T1 [][[]T1{}[0][0]]T0)
	// the element type may be accessed before it's set. Make sure we have
	// a valid type.
	if x.typ == nil {
		x.typ = Typ[Invalid]
	}

	check.index(index, length)
	return false
}

func (check *Checker) sliceExpr(x *operand, e *syntax.SliceExpr) {
	check.expr(x, e.X)
	if x.mode == invalid {
		check.use(e.Index[:]...)
		return
	}

	valid := false
	length := int64(-1) // valid if >= 0
	switch typ := optype(x.typ).(type) {
	case *Basic:
		if isString(typ) {
			if e.Full {
				check.error(x, invalidOp+"3-index slice of string")
				x.mode = invalid
				return
			}
			valid = true
			if x.mode == constant_ {
				length = int64(len(constant.StringVal(x.val)))
			}
			// spec: "For untyped string operands the result
			// is a non-constant value of type string."
			if typ.kind == UntypedString {
				x.typ = Typ[String]
			}
		}

	case *Array:
		valid = true
		length = typ.len
		if x.mode != variable {
			check.errorf(x, invalidOp+"%s (slice of unaddressable value)", x)
			x.mode = invalid
			return
		}
		x.typ = &Slice{elem: typ.elem}

	case *Pointer:
		if typ := asArray(typ.base); typ != nil {
			valid = true
			length = typ.len
			x.typ = &Slice{elem: typ.elem}
		}

	case *Slice:
		valid = true
		// x.typ doesn't change

	case *Sum, *TypeParam:
		check.error(x, "generic slice expressions not yet implemented")
		x.mode = invalid
		return
	}

	if !valid {
		check.errorf(x, invalidOp+"cannot slice %s", x)
		x.mode = invalid
		return
	}

	x.mode = value

	// spec: "Only the first index may be omitted; it defaults to 0."
	if e.Full && (e.Index[1] == nil || e.Index[2] == nil) {
		check.error(e, invalidAST+"2nd and 3rd index required in 3-index slice")
		x.mode = invalid
		return
	}

	// check indices
	var ind [3]int64
	for i, expr := range e.Index {
		x := int64(-1)
		switch {
		case expr != nil:
			// The "capacity" is only known statically for strings, arrays,
			// and pointers to arrays, and it is the same as the length for
			// those types.
			max := int64(-1)
			if length >= 0 {
				max = length + 1
			}
			if _, v := check.index(expr, max); v >= 0 {
				x = v
			}
		case i == 0:
			// default is 0 for the first index
			x = 0
		case length >= 0:
			// default is length (== capacity) otherwise
			x = length
		}
		ind[i] = x
	}

	// constant indices must be in range
	// (check.index already checks that existing indices >= 0)
L:
	for i, x := range ind[:len(ind)-1] {
		if x > 0 {
			for _, y := range ind[i+1:] {
				if y >= 0 && x > y {
					check.errorf(e, "invalid slice indices: %d > %d", x, y)
					break L // only report one error, ok to continue
				}
			}
		}
	}
}

// singleIndex returns the (single) index from the index expression e.
// If the index is missing, or if there are multiple indices, an error
// is reported and the result is nil.
func (check *Checker) singleIndex(e *syntax.IndexExpr) syntax.Expr {
	index := e.Index
	if index == nil {
		check.errorf(e, invalidAST+"missing index for %s", e.X)
		return nil
	}
	if l, _ := index.(*syntax.ListExpr); l != nil {
		if n := len(l.ElemList); n <= 1 {
			check.errorf(e, invalidAST+"invalid use of ListExpr for index expression %v with %d indices", e, n)
			return nil
		}
		// len(l.ElemList) > 1
		check.error(l.ElemList[1], invalidOp+"more than one index")
		index = l.ElemList[0] // continue with first index
	}
	return index
}

// index checks an index expression for validity.
// If max >= 0, it is the upper bound for index.
// If the result typ is != Typ[Invalid], index is valid and typ is its (possibly named) integer type.
// If the result val >= 0, index is valid and val is its constant int value.
func (check *Checker) index(index syntax.Expr, max int64) (typ Type, val int64) {
	typ = Typ[Invalid]
	val = -1

	var x operand
	check.expr(&x, index)
	if !check.isValidIndex(&x, "index", false) {
		return
	}

	if x.mode != constant_ {
		return x.typ, -1
	}

	if x.val.Kind() == constant.Unknown {
		return
	}

	v, ok := constant.Int64Val(x.val)
	assert(ok)
	if max >= 0 && v >= max {
		if check.conf.CompilerErrorMessages {
			check.errorf(&x, invalidArg+"array index %s out of bounds [0:%d]", x.val.String(), max)
		} else {
			check.errorf(&x, invalidArg+"index %s is out of bounds", &x)
		}
		return
	}

	// 0 <= v [ && v < max ]
	return x.typ, v
}

// isValidIndex checks whether operand x satisfies the criteria for integer
// index values. If allowNegative is set, a constant operand may be negative.
// If the operand is not valid, an error is reported (using what as context)
// and the result is false.
func (check *Checker) isValidIndex(x *operand, what string, allowNegative bool) bool {
	if x.mode == invalid {
		return false
	}

	// spec: "a constant index that is untyped is given type int"
	check.convertUntyped(x, Typ[Int])
	if x.mode == invalid {
		return false
	}

	// spec: "the index x must be of integer type or an untyped constant"
	if !isInteger(x.typ) {
		check.errorf(x, invalidArg+"%s %s must be integer", what, x)
		return false
	}

	if x.mode == constant_ {
		// spec: "a constant index must be non-negative ..."
		if !allowNegative && constant.Sign(x.val) < 0 {
			check.errorf(x, invalidArg+"%s %s must not be negative", what, x)
			return false
		}

		// spec: "... and representable by a value of type int"
		if !representableConst(x.val, check, Typ[Int], &x.val) {
			check.errorf(x, invalidArg+"%s %s overflows int", what, x)
			return false
		}
	}

	return true
}

// indexElts checks the elements (elts) of an array or slice composite literal
// against the literal's element type (typ), and the element indices against
// the literal length if known (length >= 0). It returns the length of the
// literal (maximum index value + 1).
func (check *Checker) indexedElts(elts []syntax.Expr, typ Type, length int64) int64 {
	visited := make(map[int64]bool, len(elts))
	var index, max int64
	for _, e := range elts {
		// determine and check index
		validIndex := false
		eval := e
		if kv, _ := e.(*syntax.KeyValueExpr); kv != nil {
			if typ, i := check.index(kv.Key, length); typ != Typ[Invalid] {
				if i >= 0 {
					index = i
					validIndex = true
				} else {
					check.errorf(e, "index %s must be integer constant", kv.Key)
				}
			}
			eval = kv.Value
		} else if length >= 0 && index >= length {
			check.errorf(e, "index %d is out of bounds (>= %d)", index, length)
		} else {
			validIndex = true
		}

		// if we have a valid index, check for duplicate entries
		if validIndex {
			if visited[index] {
				check.errorf(e, "duplicate index %d in array or slice literal", index)
			}
			visited[index] = true
		}
		index++
		if index > max {
			max = index
		}

		// check element against composite literal element type
		var x operand
		check.exprWithHint(&x, eval, typ)
		check.assignment(&x, typ, "array or slice literal")
	}
	return max
}
