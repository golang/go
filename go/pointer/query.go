package pointer

import (
	"errors"
	"fmt"
	"go/ast"
	"go/parser"
	"go/token"
	"go/types"
	"strconv"
)

// An extendedQuery represents a sequence of destructuring operations
// applied to an ssa.Value (denoted by "x").
type extendedQuery struct {
	ops []interface{}
	ptr *Pointer
}

// indexValue returns the value of an integer literal used as an
// index.
func indexValue(expr ast.Expr) (int, error) {
	lit, ok := expr.(*ast.BasicLit)
	if !ok {
		return 0, fmt.Errorf("non-integer index (%T)", expr)
	}
	if lit.Kind != token.INT {
		return 0, fmt.Errorf("non-integer index %s", lit.Value)
	}
	return strconv.Atoi(lit.Value)
}

// parseExtendedQuery parses and validates a destructuring Go
// expression and returns the sequence of destructuring operations.
// See parseDestructuringExpr for details.
func parseExtendedQuery(typ types.Type, query string) ([]interface{}, types.Type, error) {
	expr, err := parser.ParseExpr(query)
	if err != nil {
		return nil, nil, err
	}
	ops, typ, err := destructuringOps(typ, expr)
	if err != nil {
		return nil, nil, err
	}
	if len(ops) == 0 {
		return nil, nil, errors.New("invalid query: must not be empty")
	}
	if ops[0] != "x" {
		return nil, nil, fmt.Errorf("invalid query: query operand must be named x")
	}
	if !CanPoint(typ) {
		return nil, nil, fmt.Errorf("query does not describe a pointer-like value: %s", typ)
	}
	return ops, typ, nil
}

// destructuringOps parses a Go expression consisting only of an
// identifier "x", field selections, indexing, channel receives, load
// operations and parens---for example: "<-(*x[i])[key]"--- and
// returns the sequence of destructuring operations on x.
func destructuringOps(typ types.Type, expr ast.Expr) ([]interface{}, types.Type, error) {
	switch expr := expr.(type) {
	case *ast.SelectorExpr:
		out, typ, err := destructuringOps(typ, expr.X)
		if err != nil {
			return nil, nil, err
		}

		var structT *types.Struct
		switch typ := typ.Underlying().(type) {
		case *types.Pointer:
			var ok bool
			structT, ok = typ.Elem().Underlying().(*types.Struct)
			if !ok {
				return nil, nil, fmt.Errorf("cannot access field %s of pointer to type %s", expr.Sel.Name, typ.Elem())
			}

			out = append(out, "load")
		case *types.Struct:
			structT = typ
		default:
			return nil, nil, fmt.Errorf("cannot access field %s of type %s", expr.Sel.Name, typ)
		}

		for i := 0; i < structT.NumFields(); i++ {
			field := structT.Field(i)
			if field.Name() == expr.Sel.Name {
				out = append(out, "field", i)
				return out, field.Type().Underlying(), nil
			}
		}
		// TODO(dh): supporting embedding would need something like
		// types.LookupFieldOrMethod, but without taking package
		// boundaries into account, because we may want to access
		// unexported fields. If we were only interested in one level
		// of unexported name, we could determine the appropriate
		// package and run LookupFieldOrMethod with that. However, a
		// single query may want to cross multiple package boundaries,
		// and at this point it's not really worth the complexity.
		return nil, nil, fmt.Errorf("no field %s in %s (embedded fields must be resolved manually)", expr.Sel.Name, structT)
	case *ast.Ident:
		return []interface{}{expr.Name}, typ, nil
	case *ast.BasicLit:
		return []interface{}{expr.Value}, nil, nil
	case *ast.IndexExpr:
		out, typ, err := destructuringOps(typ, expr.X)
		if err != nil {
			return nil, nil, err
		}
		switch typ := typ.Underlying().(type) {
		case *types.Array:
			out = append(out, "arrayelem")
			return out, typ.Elem().Underlying(), nil
		case *types.Slice:
			out = append(out, "sliceelem")
			return out, typ.Elem().Underlying(), nil
		case *types.Map:
			out = append(out, "mapelem")
			return out, typ.Elem().Underlying(), nil
		case *types.Tuple:
			out = append(out, "index")
			idx, err := indexValue(expr.Index)
			if err != nil {
				return nil, nil, err
			}
			out = append(out, idx)
			if idx >= typ.Len() || idx < 0 {
				return nil, nil, fmt.Errorf("tuple index %d out of bounds", idx)
			}
			return out, typ.At(idx).Type().Underlying(), nil
		default:
			return nil, nil, fmt.Errorf("cannot index type %s", typ)
		}

	case *ast.UnaryExpr:
		if expr.Op != token.ARROW {
			return nil, nil, fmt.Errorf("unsupported unary operator %s", expr.Op)
		}
		out, typ, err := destructuringOps(typ, expr.X)
		if err != nil {
			return nil, nil, err
		}
		ch, ok := typ.(*types.Chan)
		if !ok {
			return nil, nil, fmt.Errorf("cannot receive from value of type %s", typ)
		}
		out = append(out, "recv")
		return out, ch.Elem().Underlying(), err
	case *ast.ParenExpr:
		return destructuringOps(typ, expr.X)
	case *ast.StarExpr:
		out, typ, err := destructuringOps(typ, expr.X)
		if err != nil {
			return nil, nil, err
		}
		ptr, ok := typ.(*types.Pointer)
		if !ok {
			return nil, nil, fmt.Errorf("cannot dereference type %s", typ)
		}
		out = append(out, "load")
		return out, ptr.Elem().Underlying(), err
	default:
		return nil, nil, fmt.Errorf("unsupported expression %T", expr)
	}
}

func (a *analysis) evalExtendedQuery(t types.Type, id nodeid, ops []interface{}) (types.Type, nodeid) {
	pid := id
	// TODO(dh): we're allocating intermediary nodes each time
	// evalExtendedQuery is called. We should probably only generate
	// them once per (v, ops) pair.
	for i := 1; i < len(ops); i++ {
		var nid nodeid
		switch ops[i] {
		case "recv":
			t = t.(*types.Chan).Elem().Underlying()
			nid = a.addNodes(t, "query.extended")
			a.load(nid, pid, 0, a.sizeof(t))
		case "field":
			i++ // fetch field index
			tt := t.(*types.Struct)
			idx := ops[i].(int)
			offset := a.offsetOf(t, idx)
			t = tt.Field(idx).Type().Underlying()
			nid = a.addNodes(t, "query.extended")
			a.copy(nid, pid+nodeid(offset), a.sizeof(t))
		case "arrayelem":
			t = t.(*types.Array).Elem().Underlying()
			nid = a.addNodes(t, "query.extended")
			a.copy(nid, 1+pid, a.sizeof(t))
		case "sliceelem":
			t = t.(*types.Slice).Elem().Underlying()
			nid = a.addNodes(t, "query.extended")
			a.load(nid, pid, 1, a.sizeof(t))
		case "mapelem":
			tt := t.(*types.Map)
			t = tt.Elem()
			ksize := a.sizeof(tt.Key())
			vsize := a.sizeof(tt.Elem())
			nid = a.addNodes(t, "query.extended")
			a.load(nid, pid, ksize, vsize)
		case "index":
			i++ // fetch index
			tt := t.(*types.Tuple)
			idx := ops[i].(int)
			t = tt.At(idx).Type().Underlying()
			nid = a.addNodes(t, "query.extended")
			a.copy(nid, pid+nodeid(idx), a.sizeof(t))
		case "load":
			t = t.(*types.Pointer).Elem().Underlying()
			nid = a.addNodes(t, "query.extended")
			a.load(nid, pid, 0, a.sizeof(t))
		default:
			// shouldn't happen
			panic(fmt.Sprintf("unknown op %q", ops[i]))
		}
		pid = nid
	}

	return t, pid
}
