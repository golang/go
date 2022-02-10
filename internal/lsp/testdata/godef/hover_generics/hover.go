package hover

type value[T any] struct { //@mark(value, "value"),hoverdef("value", value)
	val T   //@mark(valueTparam, "T"),hoverdef("T", valueTparam)
	Q   int //@mark(valueQfield, "Q"),hoverdef("Q", valueQfield)
}

type Value[T any] struct {
	val T   //@mark(ValueTparam, "T"),hoverdef("T", ValueTparam)
	Q   int //@mark(ValueQfield, "Q"),hoverdef("Q", ValueQfield)
}
