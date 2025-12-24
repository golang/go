package noder

import (
	"cmd/compile/internal/syntax"
	"cmd/compile/internal/types2"
)

// collectEnumLoweringHints walks all parsed files (which have been type-checked once by types2)
// and collects shape-based storage hints for enum lowering:
// - maxStack: max payload size among pointer-free variant payloads observed
// - needHeap: whether any observed variant payload contains pointers
//
// Key is the base enum type name (unqualified).
func collectEnumLoweringHints(noders []*noder) map[string]syntax.EnumLoweringHint {
	hints := make(map[string]syntax.EnumLoweringHint)

	for _, p := range noders {
		if p == nil || p.file == nil {
			continue
		}
		syntax.Inspect(p.file, func(n syntax.Node) bool {
			// We rely on types2 having populated type info in syntax nodes (StoreTypesInSyntax=true).
			// Collect from any expression whose type is an instantiated enum.
			e, ok := n.(syntax.Expr)
			if !ok || e == nil {
				return true
			}
			tv := e.GetTypeInfo()
			if tv.Type == nil {
				return true
			}

			// Look for an underlying *types2.Enum.
			ut := types2.Unalias(tv.Type).Underlying()
			enum, ok := ut.(*types2.Enum)
			if !ok || enum == nil {
				return true
			}

			// Base name for this enum type: try Named.Obj().Name().
			// If not a named type, we can't safely key it.
			named, _ := types2.Unalias(tv.Type).(*types2.Named)
			if named == nil || named.Obj() == nil {
				return true
			}
			base := named.Obj().Name()
			if base == "" {
				return true
			}

			cur := hints[base]

			// Scan per-variant layout (computed in types2).
			for i := 0; i < enum.NumVariants(); i++ {
				v := enum.Variant(i)
				if v == nil {
					continue
				}
				// Query by name to avoid relying on slice alignment.
				sz, hasPtr, ok := enum.VariantLayoutInfo(v.Name())
				if !ok {
					continue
				}
				if hasPtr {
					cur.NeedHeap = true
				} else {
					if int(sz) > cur.MaxStack {
						cur.MaxStack = int(sz)
					}
				}
			}

			hints[base] = cur
			return true
		})
	}

	return hints
}


