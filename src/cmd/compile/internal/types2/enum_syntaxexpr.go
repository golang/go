package types2

import (
	"cmd/compile/internal/syntax"
	"strconv"
)

// VariantPayloadSyntax is a duck-typed helper used by syntax-lowering code to
// obtain a concrete syntax.Expr for a variant payload type, after type inference.
// It returns nil for unit variants.
func (t *Enum) VariantPayloadSyntax(variantName string, pos syntax.Pos) syntax.Expr {
	if t == nil {
		return nil
	}
	for _, v := range t.variants {
		if v != nil && v.name == variantName {
			// Unit variant payload modeled as empty struct.
			if st, _ := v.typ.Underlying().(*Struct); st != nil && st.NumFields() == 0 {
				return nil
			}
			return typeToSyntaxExpr(pos, v.typ)
		}
	}
	return nil
}

// typeToSyntaxExpr converts a (sufficiently concrete) types2.Type into a syntax type expression.
// This is intentionally partial, but covers all enum payload shapes we generate.
func typeToSyntaxExpr(pos syntax.Pos, t Type) syntax.Expr {
	if t == nil {
		return nil
	}
	t = Unalias(t)
	switch tt := t.(type) {
	case *Basic:
		n := syntax.NewName(pos, tt.name)
		return n
	case *TypeParam:
		return syntax.NewName(pos, tt.obj.name)
	case *Pointer:
		op := new(syntax.Operation)
		op.SetPos(pos)
		op.Op = syntax.Mul
		op.X = typeToSyntaxExpr(pos, tt.base)
		return op
	case *Slice:
		st := new(syntax.SliceType)
		st.SetPos(pos)
		st.Elem = typeToSyntaxExpr(pos, tt.elem)
		return st
	case *Array:
		at := new(syntax.ArrayType)
		at.SetPos(pos)
		at.Elem = typeToSyntaxExpr(pos, tt.elem)
		// length is an int64 constant in types2
		lit := new(syntax.BasicLit)
		lit.SetPos(pos)
		lit.Kind = syntax.IntLit
		lit.Value = strconv.FormatInt(tt.len, 10)
		at.Len = lit
		return at
	case *Map:
		mt := new(syntax.MapType)
		mt.SetPos(pos)
		mt.Key = typeToSyntaxExpr(pos, tt.key)
		mt.Value = typeToSyntaxExpr(pos, tt.elem)
		return mt
	case *Chan:
		ct := new(syntax.ChanType)
		ct.SetPos(pos)
		ct.Elem = typeToSyntaxExpr(pos, tt.elem)
		// ignore direction for now
		return ct
	case *Struct:
		st := new(syntax.StructType)
		st.SetPos(pos)
		for i := 0; i < tt.NumFields(); i++ {
			f := tt.Field(i)
			sf := new(syntax.Field)
			sf.SetPos(pos)
			if f.name != "" {
				sf.Name = syntax.NewName(pos, f.name)
			}
			sf.Type = typeToSyntaxExpr(pos, f.typ)
			st.FieldList = append(st.FieldList, sf)
		}
		return st
	case *Tuple:
		// Convert tuple (T1, T2, ...) into a ListExpr
		if tt.Len() == 0 {
			return nil
		}
		if tt.Len() == 1 {
			// Single element: return as-is (not wrapped in ListExpr)
			return typeToSyntaxExpr(pos, tt.At(0).typ)
		}
		// Multiple elements: return as ListExpr
		list := make([]syntax.Expr, tt.Len())
		for i := 0; i < tt.Len(); i++ {
			list[i] = typeToSyntaxExpr(pos, tt.At(i).typ)
		}
		l := new(syntax.ListExpr)
		l.SetPos(pos)
		l.ElemList = list
		return l
	case *Named:
		// Build a name (best-effort). If this is an instantiated named type,
		// preserve its type arguments so syntax-lowering can generate concrete
		// payload types like Option[int] rather than the uninstantiated Option.
		if tt.obj != nil {
			base := syntax.NewName(pos, tt.obj.name)
			if ta := tt.TypeArgs(); ta != nil && ta.Len() > 0 {
				var idx syntax.Expr
				if ta.Len() == 1 {
					idx = typeToSyntaxExpr(pos, ta.At(0))
				} else {
					l := new(syntax.ListExpr)
					l.SetPos(pos)
					l.ElemList = make([]syntax.Expr, 0, ta.Len())
					for i := 0; i < ta.Len(); i++ {
						l.ElemList = append(l.ElemList, typeToSyntaxExpr(pos, ta.At(i)))
					}
					idx = l
				}
				ix := new(syntax.IndexExpr)
				ix.SetPos(pos)
				ix.X = base
				ix.Index = idx
				return ix
			}
			return base
		}
		return syntax.NewName(pos, "/*named*/")
	default:
		// Fallback: use string form as a Name (best-effort).
		return syntax.NewName(pos, t.String())
	}
}


