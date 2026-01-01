package types2

// Enum represents an enum type.
type Enum struct {
	variants []*Var
	// variantHasPayload indicates whether a variant was declared with an explicit payload.
	// It is aligned with variants slice.
	variantHasPayload []bool

	// layout info for enum payload storage.
	// For a generic (uninstantiated) enum, these may be conservative/unknown.
	// For an instantiated enum (after substitution), these should reflect the
	// concrete payload types.
	maxPayloadSize int64
	hasPointers    bool

	// per-variant layout (aligned with variants slice)
	variantSize     []int64
	variantHasPtr   []bool
}

// NewEnum returns a new Enum type.
func NewEnum(variants []*Var) *Enum {
	return &Enum{variants: variants}
}

func (t *Enum) NumVariants() int {
	return len(t.variants)
}

func (t *Enum) Variant(i int) *Var {
	return t.variants[i]
}

func (t *Enum) MaxPayloadSize() int64 { return t.maxPayloadSize }
func (t *Enum) HasPointers() bool     { return t.hasPointers }

// VariantLayoutInfo is a duck-typed helper used by syntax-lowering code (which
// cannot import types2) to decide whether a variant payload must go to heap.
func (t *Enum) VariantLayoutInfo(variantName string) (size int64, hasPtr bool, ok bool) {
	if t == nil {
		return 0, false, false
	}
	for i, v := range t.variants {
		if v != nil && v.name == variantName {
			if i < len(t.variantSize) && i < len(t.variantHasPtr) {
				return t.variantSize[i], t.variantHasPtr[i], true
			}
			// Fallback if per-variant slices weren't computed.
			return 0, typeHasPointers(v.typ), true
		}
	}
	return 0, false, false
}

func (t *Enum) Underlying() Type { return t }
func (t *Enum) String() string   { return TypeString(t, nil) }
