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
	for i, v := range e.variants {
		if v == nil {
			continue
		}
		t := v.typ
		if t == nil {
			continue
		}
		if containsTypeParam(t) {
			// unknown until instantiation
			if i < len(e.variantHasPtr) {
				e.variantHasPtr[i] = true
			}
			return 0, true
		}
		if i < len(e.variantHasPtr) {
			e.variantHasPtr[i] = typeHasPointers(t)
		}
		if check != nil && check.conf.Sizes != nil {
			sz := check.conf.Sizes.Sizeof(t)
			if i < len(e.variantSize) {
				e.variantSize[i] = sz
			}
			if sz > maxSize {
				maxSize = sz
			}
		}
		if typeHasPointers(t) {
			hasPtr = true
		}
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

// typeHasPointers is a conservative pointer-contains check for types2 types.
func typeHasPointers(t Type) bool {
	switch tt := Unalias(t).Underlying().(type) {
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
		return typeHasPointers(tt.elem)
	case *Struct:
		for _, f := range tt.fields {
			if f != nil && typeHasPointers(f.typ) {
				return true
			}
		}
		return false
	case *Tuple:
		for _, v := range tt.vars {
			if v != nil && typeHasPointers(v.typ) {
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
			if v != nil && typeHasPointers(v.typ) {
				return true
			}
		}
		return false
	case *Named:
		// Underlying already handled, but keep conservative.
		return typeHasPointers(tt.Underlying())
	default:
		return true
	}
}


