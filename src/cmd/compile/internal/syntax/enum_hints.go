package syntax

// EnumLoweringHint is populated by the noder pipeline (after the first types2 pass)
// and consumed by enum lowering in syntax/enum.go and syntax/rewrite.go.
//
// MaxStack is the maximum payload size (bytes) among pointer-free variant payloads
// observed in this package for this enum (across all instantiations/shapes).
// NeedHeap reports whether any observed variant payload contains pointers.
type EnumLoweringHint struct {
	MaxStack int
	NeedHeap bool
}

var enumLoweringHints map[string]EnumLoweringHint

// SetEnumLoweringHints sets package-local enum lowering hints for the current compilation unit.
// The map key is the base enum type name (e.g. "Option", "Result").
func SetEnumLoweringHints(h map[string]EnumLoweringHint) {
	enumLoweringHints = h
}

func getEnumLoweringHint(enumName string) (EnumLoweringHint, bool) {
	if enumLoweringHints == nil {
		return EnumLoweringHint{}, false
	}
	h, ok := enumLoweringHints[enumName]
	return h, ok
}


