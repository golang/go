package builtins

func _() {
	const (
		foo = iota //@complete(" //", iota)
	)

	iota //@complete(" //")

	var iota int //@item(iotaVar, "iota", "int", "var")

	iota //@complete(" //", iotaVar)
}

func _() {
	var twoRedUpEnd bool //@item(TRUEVar, "twoRedUpEnd", "bool", "var")

	var _ bool = true //@rank(" //", _true, TRUEVar)
}
