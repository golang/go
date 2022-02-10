package hover

type value[T any] struct { //@mark(value, "value"),hoverdef("value", value)
	val T //@mark(Tparam, "T"),hoverdef("T", Tparam)
	q   int
}
