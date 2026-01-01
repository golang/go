package syntax

import (
	"sort"
	"unicode"
)

// RewriteTypeAlgebra rewrites MyGo "type algebra" expressions in type positions:
//   - Sum types:     A + B   =>  enum { A(A); B(B) }   (order canonicalized)
//   - Product types: A * B   =>  left as Operation(Mul, X, Y) for types2 to interpret
//
// This pass runs before the first types2 pass (see rewrite_pretypes2.go).
func RewriteTypeAlgebra(file *File) {
	if file == nil {
		return
	}
	for _, decl := range file.DeclList {
		switch d := decl.(type) {
		case *TypeDecl:
			for _, tp := range d.TParamList {
				rewriteFieldType(tp)
			}
			d.Type = rewriteTypeExpr(d.Type)
		case *ConstDecl:
			d.Type = rewriteTypeExpr(d.Type)
		case *VarDecl:
			d.Type = rewriteTypeExpr(d.Type)
		case *FuncDecl:
			rewriteFieldType(d.Recv)
			for _, tp := range d.TParamList {
				rewriteFieldType(tp)
			}
			rewriteFuncType(d.Type)
		}
	}
}

func rewriteFieldType(f *Field) {
	if f == nil {
		return
	}
	f.Type = rewriteTypeExpr(f.Type)
}

func rewriteFuncType(t *FuncType) {
	if t == nil {
		return
	}
	for _, p := range t.ParamList {
		rewriteFieldType(p)
	}
	for _, r := range t.ResultList {
		rewriteFieldType(r)
	}
}

func rewriteTypeExpr(e Expr) Expr {
	if e == nil {
		return nil
	}

	switch x := e.(type) {
	case *Operation:
		x.X = rewriteTypeExpr(x.X)
		if x.Y != nil {
			x.Y = rewriteTypeExpr(x.Y)
		}
		// Sum type: A + B  => enum { ... }
		if x.Op == Add && x.Y != nil {
			return rewriteSumType(x)
		}
		return x

	case *ParenExpr:
		x.X = rewriteTypeExpr(x.X)
		return x

	case *IndexExpr:
		x.X = rewriteTypeExpr(x.X)
		x.Index = rewriteTypeExpr(x.Index)
		return x

	case *SelectorExpr:
		// Qualified type name: pkg.Type. Keep as-is.
		// (x.X is a package expression; not a type expression).
		return x

	case *ListExpr:
		for i := range x.ElemList {
			x.ElemList[i] = rewriteTypeExpr(x.ElemList[i])
		}
		return x

	case *ArrayType:
		x.Elem = rewriteTypeExpr(x.Elem)
		return x

	case *SliceType:
		x.Elem = rewriteTypeExpr(x.Elem)
		return x

	case *DotsType:
		x.Elem = rewriteTypeExpr(x.Elem)
		return x

	case *MapType:
		x.Key = rewriteTypeExpr(x.Key)
		x.Value = rewriteTypeExpr(x.Value)
		return x

	case *ChanType:
		x.Elem = rewriteTypeExpr(x.Elem)
		return x

	case *StructType:
		for _, f := range x.FieldList {
			rewriteFieldType(f)
		}
		return x

	case *InterfaceType:
		for _, f := range x.MethodList {
			if f == nil {
				continue
			}
			// f.Type may be a signature (*FuncType) or an embedded element (type expr).
			f.Type = rewriteTypeExpr(f.Type)
		}
		return x

	case *FuncType:
		rewriteFuncType(x)
		return x

	case *EnumType:
		for _, v := range x.VariantList {
			if v == nil {
				continue
			}
			v.Type = rewriteTypeExpr(v.Type)
		}
		return x

	default:
		return e
	}
}

type sumOperand struct {
	key       string // canonical sort key (printed type)
	variant   string // enum variant name (identifier-ish)
	typ       Expr   // payload type; nil means unit variant (nil)
	isUnitNil bool
	pos       Pos
}

func rewriteSumType(op *Operation) Expr {
	ops := flattenSumOperands(op)
	if len(ops) == 0 {
		return op
	}

	// Normalize and sort for commutativity (structural type).
	items := make([]sumOperand, 0, len(ops))
	seenKey := make(map[string]bool)
	seenVariant := make(map[string]bool)
	for _, t := range ops {
		if t == nil {
			continue
		}
		pos := t.Pos()
		key := String(t)
		if seenKey[key] {
			// Redundant: drop duplicates (keeps stable ordering after sort).
			continue
		}
		seenKey[key] = true

		item := sumOperand{key: key, pos: pos}
		if n, ok := t.(*Name); ok && n.Value == "nil" {
			item.isUnitNil = true
			item.typ = nil
			item.variant = "nil"
		} else {
			item.typ = t
			item.variant = variantNameFromTypeExpr(t)
		}
		if seenVariant[item.variant] {
			// Avoid generating an enum with duplicate variant names (would break ctor sugar).
			// Keep it deterministic by suffixing.
			i := 2
			for {
				v := item.variant + "_" + itoa(i)
				if !seenVariant[v] {
					item.variant = v
					break
				}
				i++
			}
		}
		seenVariant[item.variant] = true
		items = append(items, item)
	}

	sort.Slice(items, func(i, j int) bool {
		return items[i].key < items[j].key
	})

	et := new(EnumType)
	et.SetPos(op.Pos())
	for _, it := range items {
		v := new(EnumVariant)
		v.SetPos(it.pos)
		v.Name = NewName(it.pos, it.variant)
		if !it.isUnitNil {
			v.Type = it.typ
		}
		et.VariantList = append(et.VariantList, v)
	}
	return et
}

func flattenSumOperands(e Expr) []Expr {
	var out []Expr
	var walk func(x Expr)
	walk = func(x Expr) {
		if x == nil {
			return
		}
		if op, ok := x.(*Operation); ok && op.Op == Add && op.Y != nil {
			walk(op.X)
			walk(op.Y)
			return
		}
		out = append(out, x)
	}
	walk(e)
	return out
}

func variantNameFromTypeExpr(t Expr) string {
	if t == nil {
		return "_"
	}
	// Prefer simple identifiers when possible for nicer ctor syntax.
	switch x := t.(type) {
	case *Name:
		return sanitizeIdent(x.Value)
	case *SelectorExpr:
		if x.Sel != nil {
			return sanitizeIdent(String(x))
		}
	case *IndexExpr:
		return sanitizeIdent(String(x))
	}
	return sanitizeIdent(String(t))
}

func sanitizeIdent(s string) string {
	// Convert printed type to a selector-friendly identifier:
	// keep letters/digits/underscore; everything else => underscore.
	var out []rune
	lastUnderscore := false
	for _, r := range s {
		ok := r == '_' || unicode.IsLetter(r) || unicode.IsDigit(r)
		if !ok {
			if !lastUnderscore {
				out = append(out, '_')
				lastUnderscore = true
			}
			continue
		}
		if r == '_' {
			if lastUnderscore {
				continue
			}
			lastUnderscore = true
			out = append(out, r)
			continue
		}
		lastUnderscore = false
		out = append(out, r)
	}
	if len(out) == 0 {
		return "_"
	}
	// Trim trailing underscore(s).
	for len(out) > 0 && out[len(out)-1] == '_' {
		out = out[:len(out)-1]
	}
	if len(out) == 0 {
		return "_"
	}
	// Identifiers can't start with a digit.
	if unicode.IsDigit(out[0]) {
		out = append([]rune{'_'}, out...)
	}
	return string(out)
}

func itoa(n int) string {
	// small helper, avoids importing strconv for this file.
	if n == 0 {
		return "0"
	}
	var buf [32]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	return string(buf[i:])
}
