package types2

import "cmd/compile/internal/syntax"

// asEnum returns the underlying *Enum for typ, if any.
func asEnumType(typ Type) (*Enum, bool) {
	if typ == nil {
		return nil, false
	}
	switch t := Unalias(typ).Underlying().(type) {
	case *Enum:
		return t, true
	default:
		return nil, false
	}
}

// enumVariantByName returns the variant *Var and true if found.
func enumVariantByName(e *Enum, name string) (*Var, bool) {
	if e == nil {
		return nil, false
	}
	for _, v := range e.variants {
		if v != nil && v.name == name {
			return v, true
		}
	}
	return nil, false
}

type enumCasePattern struct {
	variantName string
	// bindNames are the identifiers in the pattern args (excluding blanks).
	// For tuple payloads, bindNames aligns with tuple elements; for scalar payloads, it has length 1.
	bindNames []*syntax.Name
	// bindTypes aligns with bindNames (same length).
	bindTypes []Type
}

// parseEnumCasePattern attempts to interpret e as an enum pattern for the given enum type.
// It recognizes:
//   - Enum.Variant          (unit variant)
//   - Enum.Variant(x)       (scalar payload)
//   - Enum.Variant(x, y...) (tuple payload)
//
// It does not typecheck literals; that is handled elsewhere.
func (check *Checker) parseEnumCasePattern(enum *Enum, e syntax.Expr) (enumCasePattern, bool) {
	// Peel list (handled by caller).
	var pat enumCasePattern
	if e == nil || enum == nil {
		return pat, false
	}

	var sel *syntax.SelectorExpr
	var args []syntax.Expr
	switch x := syntax.Unparen(e).(type) {
	case *syntax.SelectorExpr:
		sel = x
	case *syntax.CallExpr:
		if s, ok := syntax.Unparen(x.Fun).(*syntax.SelectorExpr); ok {
			sel = s
			args = x.ArgList
		}
	default:
		return pat, false
	}
	if sel == nil || sel.Sel == nil {
		return pat, false
	}

	vname := sel.Sel.Value
	v, ok := enumVariantByName(enum, vname)
	if !ok || v == nil {
		return pat, false
	}
	pat.variantName = vname

	// Unit variant: accept SelectorExpr with no args.
	// Payload variant: must be CallExpr; handled by caller (we return ok=true but no bindings if no args).
	payload := v.typ
	if payload == nil {
		return pat, true
	}

	// Treat struct payload (tuple) specially.
	if st, _ := payload.Underlying().(*Struct); st != nil && st.NumFields() > 0 && len(args) > 0 {
		// Tuple payload: bind names correspond to struct fields _0.._n.
		n := st.NumFields()
		// If arg count mismatches, still return ok=true and let checker report in switch logic.
		if len(args) > n {
			n = len(args)
		}
		for i := 0; i < len(args) && i < st.NumFields(); i++ {
			if id, ok := args[i].(*syntax.Name); ok && id != nil && id.Value != "_" {
				pat.bindNames = append(pat.bindNames, id)
				pat.bindTypes = append(pat.bindTypes, st.Field(i).typ)
			}
		}
		return pat, true
	}

	// Scalar payload: single binding if first arg is an identifier.
	if len(args) >= 1 {
		if id, ok := args[0].(*syntax.Name); ok && id != nil && id.Value != "_" {
			pat.bindNames = []*syntax.Name{id}
			pat.bindTypes = []Type{payload}
		}
	}
	return pat, true
}


