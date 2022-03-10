package reflect

// Not an actual implementation of DeepEqual. This is a model that supports
// the bare minimum needed to get through testing interp.
//
// Does not handle cycles.
//
// Note: unclear if reflect.go can support this.
func DeepEqual(x, y interface{}) bool {
	if x == nil || y == nil {
		return x == y
	}
	v1 := ValueOf(x)
	v2 := ValueOf(y)

	return deepValueEqual(v1, v2, make(map[visit]bool))
}

// Key for the visitedMap in deepValueEqual.
type visit struct {
	a1, a2 uintptr
	typ    Type
}

func deepValueEqual(v1, v2 Value, visited map[visit]bool) bool {
	if !v1.IsValid() || !v2.IsValid() {
		return v1.IsValid() == v2.IsValid()
	}
	if v1.Type() != v2.Type() {
		return false
	}

	// Short circuit on reference types that can lead to cycles in comparison.
	switch v1.Kind() {
	case Pointer, Map, Slice, Interface:
		k := visit{v1.Pointer(), v2.Pointer(), v1.Type()} // Not safe for moving GC.
		if visited[k] {
			// The comparison algorithm assumes that all checks in progress are true when it reencounters them.
			return true
		}
		visited[k] = true
	}

	switch v1.Kind() {
	case Array:
		for i := 0; i < v1.Len(); i++ {
			if !deepValueEqual(v1.Index(i), v2.Index(i), visited) {
				return false
			}
		}
		return true
	case Slice:
		if v1.IsNil() != v2.IsNil() {
			return false
		}
		if v1.Len() != v2.Len() {
			return false
		}
		if v1.Pointer() == v2.Pointer() {
			return true
		}
		for i := 0; i < v1.Len(); i++ {
			if !deepValueEqual(v1.Index(i), v2.Index(i), visited) {
				return false
			}
		}
		return true
	case Interface:
		if v1.IsNil() || v2.IsNil() {
			return v1.IsNil() == v2.IsNil()
		}
		return deepValueEqual(v1.Elem(), v2.Elem(), visited)
	case Ptr:
		if v1.Pointer() == v2.Pointer() {
			return true
		}
		return deepValueEqual(v1.Elem(), v2.Elem(), visited)
	case Struct:
		for i, n := 0, v1.NumField(); i < n; i++ {
			if !deepValueEqual(v1.Field(i), v2.Field(i), visited) {
				return false
			}
		}
		return true
	case Map:
		if v1.IsNil() != v2.IsNil() {
			return false
		}
		if v1.Len() != v2.Len() {
			return false
		}
		if v1.Pointer() == v2.Pointer() {
			return true
		}
		for _, k := range v1.MapKeys() {
			val1 := v1.MapIndex(k)
			val2 := v2.MapIndex(k)
			if !val1.IsValid() || !val2.IsValid() || !deepValueEqual(val1, val2, visited) {
				return false
			}
		}
		return true
	case Func:
		return v1.IsNil() && v2.IsNil()
	default:
		// Normal equality suffices
		return v1.Interface() == v2.Interface() // try interface comparison as a fallback.
	}
}
