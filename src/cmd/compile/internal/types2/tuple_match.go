package types2

import (
	"cmd/compile/internal/syntax"
	"fmt"
	"strings"
)

// unpackTupleExpr recognizes the tuple syntax produced by the parser:
//   (e1, e2, ..., eN)
// represented as ParenExpr{X: ListExpr{ElemList: ...}}.
func unpackTupleExpr(e syntax.Expr) ([]syntax.Expr, bool) {
	// NOTE: do NOT call syntax.Unparen here: tuple syntax relies on the presence of ParenExpr.
	// We allow nested parens like ((a, b)) by unwrapping ParenExpr nodes ourselves.
	pe, ok := e.(*syntax.ParenExpr)
	for ok && pe != nil {
		if le, ok := pe.X.(*syntax.ListExpr); ok && le != nil {
			if len(le.ElemList) < 2 {
				return nil, false
			}
			return le.ElemList, true
		}
		pe, ok = pe.X.(*syntax.ParenExpr)
	}
	return nil, false
}

func tupleTypeString(typs []Type) string {
	if len(typs) == 0 {
		return "()"
	}
	var sb strings.Builder
	sb.WriteByte('(')
	for i, t := range typs {
		if i > 0 {
			sb.WriteString(", ")
		}
		sb.WriteString(fmt.Sprint(t))
	}
	sb.WriteByte(')')
	return sb.String()
}

// tupleVariantComboKey creates a stable key for a combo (v0,v1,...).
func tupleVariantComboKey(vnames []string) string {
	// Use a separator unlikely to appear in identifiers.
	return strings.Join(vnames, "\x1f")
}

func tupleVariantComboString(vnames []string) string {
	if len(vnames) == 0 {
		return "()"
	}
	return "(" + strings.Join(vnames, ", ") + ")"
}


