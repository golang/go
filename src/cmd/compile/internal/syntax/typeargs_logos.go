// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package syntax

// LogosTypeArguments resolves the first argument's expression/type ambiguity.
// The caller must have established a generic target or a type position. Value
// indexing must never call this function. The canonical nodes stay in inst so
// checking and lowering use the same syntax; repeated calls leave it unchanged.
func LogosTypeArguments(inst *IndexExpr) {
	if inst.LogosTypeArgs {
		return
	}
	if list, ok := inst.Index.(*ListExpr); ok {
		list.ElemList[0] = logosTypeArgument(list.ElemList[0])
	} else {
		inst.Index = logosTypeArgument(inst.Index)
	}
	inst.LogosTypeArgs = true
}

func logosTypeArgument(e Expr) Expr {
	switch e := e.(type) {
	case *ParenExpr:
		e.X = logosTypeArgument(e.X)
	case *TryExpr:
		if _, call := Unparen(e.X).(*CallExpr); call {
			return e // a call followed by ? is still invalid propagation here
		}
		t := new(OptionType)
		t.pos = e.pos
		t.Elem = logosTypeArgument(e.X)
		return t
	case *TupleLit:
		t := new(TupleType)
		t.pos = e.pos
		for _, el := range e.Elems {
			name, typ := logosTypeArgumentField(el)
			t.Fields = append(t.Fields, &Field{Name: name, Type: typ})
		}
		return t
	case *TupleType:
		for _, f := range e.Fields {
			// Named fields were parsed with type_ and already have the right
			// option precedence, even when that parser dropped parentheses.
			if f.Name == nil {
				f.Name, f.Type = logosTypeArgumentField(f.Type)
			}
		}
	case *Operation:
		if e.Y != nil {
			return e
		}
		switch e.Op {
		case Mul:
			return logosTypeArgPrefix(logosTypeArgument(e.X), func(elem Expr) Expr {
				e.X = elem
				return e
			})
		case Recv:
			if ch := logosReceiveType(e.X); ch != nil {
				ch.pos = e.pos
				for c := ch; ; c = c.Elem.(*ChanType) {
					dir := c.Dir
					c.Dir = RecvOnly
					if dir != SendOnly {
						break
					}
				}
				return e.X
			}
		}
	case *IndexExpr:
		LogosTypeArguments(e)
	}
	return e
}

// logosTypeArgPrefix preserves type grammar precedence: *T? is (*T)?, but
// *(T?) is a pointer to an option. Parentheses stop the outward option move.
func logosTypeArgPrefix(typ Expr, prefix func(Expr) Expr) Expr {
	if opt, ok := typ.(*OptionType); ok {
		opt.Elem = logosTypeArgPrefix(opt.Elem, prefix)
		return opt
	}
	return prefix(typ)
}

// logosReceiveType recognizes a receive arrow whose channel type is hidden
// by an option. Validate the whole direction chain before changing any node.
// The parser also uses this pure check to admit a possible generic literal.
func logosReceiveType(e Expr) *ChanType {
	opt, ok := e.(*OptionType)
	if !ok {
		return nil
	}
	for {
		e = opt.Elem
		opt, ok = e.(*OptionType)
		if !ok {
			break
		}
	}
	ch, ok := e.(*ChanType)
	if !ok {
		return nil
	}
	for c := ch; ; {
		switch c.Dir {
		case RecvOnly:
			return nil
		case SendOnly:
			c, ok = c.Elem.(*ChanType)
			if !ok {
				return nil
			}
		default:
			return ch
		}
	}
}

// A named tuple field can look like multiplication, a call, or indexed
// multiplication until its surrounding tuple is known to be a type.
func logosTypeArgumentField(e Expr) (*Name, Expr) {
	switch e := e.(type) {
	case *Operation:
		if e.Op == Mul && e.Y != nil {
			if name := logosTypeArgName(e.X); name != nil {
				typ := logosTypeArgument(newIndirect(e.pos, e.Y))
				return name, logosTypeArgArrays(e.X, typ)
			}
		}
	case *CallExpr:
		if name := logosTypeArgName(e.Fun); name != nil && !e.HasDots && len(e.ArgList) != 0 {
			var typ Expr
			if len(e.ArgList) == 1 {
				typ = logosTypeArgument(e.ArgList[0])
				// The parser packages a named tuple argument list at the
				// call's opening parenthesis; it needs no extra wrapper.
				if tuple, ok := typ.(*TupleType); !ok || tuple.Pos() != e.Pos() {
					p := new(ParenExpr)
					p.pos = e.pos
					p.X = typ
					typ = p
				}
			} else {
				t := new(TupleLit)
				t.pos = e.pos
				t.Elems = e.ArgList
				typ = logosTypeArgument(t)
			}
			return name, logosTypeArgArrays(e.Fun, typ)
		}
	case *TryExpr:
		if name, typ := logosTypeArgumentField(e.X); name != nil {
			t := new(OptionType)
			t.pos = e.pos
			t.Elem = typ
			return name, t
		}
	}
	return nil, logosTypeArgument(e)
}

// logosTypeArgName accepts only a simple name with single-index suffixes.
// In a named field those suffixes are array lengths, which remain expressions.
func logosTypeArgName(e Expr) *Name {
	for {
		switch x := e.(type) {
		case *Name:
			return x
		case *IndexExpr:
			if _, list := x.Index.(*ListExpr); list {
				return nil
			}
			e = x.X
		default:
			return nil
		}
	}
}

// logosTypeArgArrays requires a prefix accepted by logosTypeArgName.
func logosTypeArgArrays(prefix, typ Expr) Expr {
	if index, ok := prefix.(*IndexExpr); ok {
		typ = logosTypeArgPrefix(typ, func(elem Expr) Expr {
			t := new(ArrayType)
			t.pos = index.pos
			t.Len = index.Index
			t.Elem = elem
			return t
		})
		return logosTypeArgArrays(index.X, typ)
	}
	return typ
}
