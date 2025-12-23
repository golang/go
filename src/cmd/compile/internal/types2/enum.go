package types2

// Enum represents an enum type.
type Enum struct {
	variants []*Var
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

func (t *Enum) Underlying() Type { return t }
func (t *Enum) String() string   { return TypeString(t, nil) }
