package hover

type value[T any] struct { //@mark(value, "value"),hoverdef("value", value),mark(valueTdecl, "T"),hoverdef("T",valueTdecl)
	val T   //@mark(valueTparam, "T"),hoverdef("T", valueTparam)
	Q   int //@mark(valueQfield, "Q"),hoverdef("Q", valueQfield)
}

type Value[T any] struct { //@mark(ValueTdecl, "T"),hoverdef("T",ValueTdecl)
	val T   //@mark(ValueTparam, "T"),hoverdef("T", ValueTparam)
	Q   int //@mark(ValueQfield, "Q"),hoverdef("Q", ValueQfield)
}

func F[P interface{ ~int | string }]() { //@mark(Pparam, "P"),hoverdef("P",Pparam)
	var _ P //@mark(Pvar, "P"),hoverdef("P",Pvar)
}
