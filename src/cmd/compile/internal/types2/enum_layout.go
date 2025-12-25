package types2

// enumLayoutFromVariants computes payload layout info for an enum.
// If a payload type contains unresolved type parameters, we conservatively
// report (maxSize=0, hasPointers=true) so downstream lowering can force heap.
func (check *Checker) enumLayoutFromVariants(e *Enum) (maxSize int64, hasPtr bool) {
	if e == nil {
		return 0, false
	}
	e.variantSize = nil
	e.variantHasPtr = nil
	if len(e.variants) > 0 {
		e.variantSize = make([]int64, len(e.variants))
		e.variantHasPtr = make([]bool, len(e.variants))
	}
	hasTypeParam := false
	hasUnknownSize := false
	for i, v := range e.variants {
		if v == nil {
			continue
		}
		t := v.typ
		if t == nil {
			continue
		}
		if containsTypeParam(t) {
			// Unknown until instantiation: be conservative for this variant, but
			// continue scanning so all variants get consistent per-variant info.
			//
			// This matters for generic enums like Result[T, E]:
			// if we returned early on the first variant (Ok(T)), we'd leave
			// Err(E) with the default hasPtr=false, which can incorrectly drive
			// lowering to use stack-based constructors/reads for pointerful E
			// (e.g. string) at instantiation sites.
			hasTypeParam = true
			if i < len(e.variantHasPtr) {
				e.variantHasPtr[i] = true
			}
			hasPtr = true
			continue
		}
		if i < len(e.variantHasPtr) {
			e.variantHasPtr[i] = typeHasPointers(t)
		}
		if check != nil && check.conf.Sizes != nil {
			// Avoid calling Sizes.Sizeof on types that (transitively) include unresolved Named types.
			// Sizeof may call under(named) which calls Named.Underlying -> resolve(), and if we're
			// already in the middle of resolving that named type (holding its mutex), we deadlock.
			if containsUnresolvedNamed(t) {
				hasUnknownSize = true
				// We still set hasPtr above for this variant via typeHasPointers(t).
				// Leave size as 0 (unknown).
				goto sizeDone
			}
			sz := check.conf.Sizes.Sizeof(t)
			if i < len(e.variantSize) {
				e.variantSize[i] = sz
			}
			if sz > maxSize {
				maxSize = sz
			}
		}
	sizeDone:
		if typeHasPointers(t) {
			hasPtr = true
		}
	}
	if hasTypeParam {
		// Payload size is not reliably known until instantiation.
		return 0, true
	}
	if hasUnknownSize {
		// We couldn't safely compute a size for at least one variant during this pass.
		// Be conservative: callers should treat size as unknown and prefer heap lowering.
		return 0, true
	}
	return maxSize, hasPtr
}

func containsTypeParam(t Type) bool {
	switch tt := Unalias(t).(type) {
	case *TypeParam:
		return true
	case *Pointer:
		return containsTypeParam(tt.base)
	case *Array:
		return containsTypeParam(tt.elem)
	case *Slice:
		return containsTypeParam(tt.elem)
	case *Map:
		return containsTypeParam(tt.key) || containsTypeParam(tt.elem)
	case *Chan:
		return containsTypeParam(tt.elem)
	case *Struct:
		for _, f := range tt.fields {
			if f != nil && containsTypeParam(f.typ) {
				return true
			}
		}
		return false
	case *Tuple:
		for _, v := range tt.vars {
			if v != nil && containsTypeParam(v.typ) {
				return true
			}
		}
		return false
	case *Signature:
		// be conservative
		if tt.params != nil && containsTypeParam(tt.params) {
			return true
		}
		if tt.results != nil && containsTypeParam(tt.results) {
			return true
		}
		return false
	case *Interface:
		// be conservative
		return false
	case *Named:
		// Don't expand; check type args.
		if tt.TypeArgs().Len() == 0 {
			return false
		}
		for _, a := range tt.TypeArgs().list() {
			if containsTypeParam(a) {
				return true
			}
		}
		return false
	default:
		return false
	}
}

// containsUnresolvedNamed reports whether t (transitively) contains a *Named that is not
// resolved yet (or is currently being resolved). This must not call Named.Underlying()
// or any operation that triggers resolve(), to avoid deadlocks during type checking.
func containsUnresolvedNamed(t Type) bool {
	seen := make(map[Type]bool)
	var rec func(Type) bool
	rec = func(x Type) bool {
		if x == nil {
			return false
		}
		if seen[x] {
			return false
		}
		seen[x] = true

		switch tt := Unalias(x).(type) {
		case *Named:
			if tt == nil {
				return false
			}
			// If unresolved (or in-progress), treat as unresolved.
			if tt.state() < resolved || tt.underlying == nil {
				return true
			}
			// Recurse into the already-known underlying without triggering resolve.
			return rec(tt.underlying)
		case *Pointer:
			return rec(tt.base)
		case *Array:
			return rec(tt.elem)
		case *Slice:
			return rec(tt.elem)
		case *Map:
			return rec(tt.key) || rec(tt.elem)
		case *Chan:
			return rec(tt.elem)
		case *Struct:
			for _, f := range tt.fields {
				if f != nil && rec(f.typ) {
					return true
				}
			}
			return false
		case *Tuple:
			for _, v := range tt.vars {
				if v != nil && rec(v.typ) {
					return true
				}
			}
			return false
		case *Signature:
			// be conservative; still avoid resolve-triggering paths
			if tt.params != nil && rec(tt.params) {
				return true
			}
			if tt.results != nil && rec(tt.results) {
				return true
			}
			return false
		case *Enum:
			for _, v := range tt.variants {
				if v != nil && rec(v.typ) {
					return true
				}
			}
			return false
		default:
			return false
		}
	}
	return rec(t)
}

// typeHasPointers is a conservative pointer-contains check for types2 types.
func typeHasPointers(t Type) bool {
	seen := make(map[Type]bool)
	return typeHasPointersImpl(t, seen)
}

func typeHasPointersImpl(t Type, seen map[Type]bool) bool {
	// Prevent infinite recursion for recursive types (e.g., enum List { Cons(int, List); Nil })
	if seen[t] {
		// Conservative: assume recursive types may contain pointers
		// (in practice, if it's a direct enum recursion like List -> List, it's a value cycle,
		// but safer to return true to avoid stack overflow)
		return true
	}
	seen[t] = true

	// IMPORTANT: Avoid calling Underlying() on *Named here.
	// Named.Underlying() triggers resolve(), which takes a mutex. During type-checking,
	// we may already be inside Named.resolve (holding that mutex) while computing enum layout,
	// and re-entering resolve would deadlock.
	u := Unalias(t)
	if n, ok := u.(*Named); ok && n != nil {
		// Avoid resolve() entirely: consult the already-populated underlying field.
		// If it's not available yet, be conservative.
		if n.underlying == nil {
			return true
		}
		return typeHasPointersImpl(Unalias(n.underlying), seen)
	}

	switch tt := u.Underlying().(type) {
	case *Basic:
		switch tt.kind {
		case String, UnsafePointer:
			return true
		default:
			return false
		}
	case *Pointer, *Slice, *Map, *Chan, *Signature, *Interface:
		return true
	case *Array:
		return typeHasPointersImpl(tt.elem, seen)
	case *Struct:
		for _, f := range tt.fields {
			if f != nil && typeHasPointersImpl(f.typ, seen) {
				return true
			}
		}
		return false
	case *Tuple:
		for _, v := range tt.vars {
			if v != nil && typeHasPointersImpl(v.typ, seen) {
				return true
			}
		}
		return false
	case *TypeParam:
		// unknown -> assume pointers possible
		return true
	case *Enum:
		// payload may contain pointers
		for _, v := range tt.variants {
			if v != nil && typeHasPointersImpl(v.typ, seen) {
				return true
			}
		}
		return false
	default:
		return true
	}
}
