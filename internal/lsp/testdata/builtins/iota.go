package builtins

func _() {
	const (
		foo = iota //@complete(" //", iota)
	)

	iota //@complete(" //")

	var iota int //@item(iotaVar, "iota", "int", "var")

	iota //@complete(" //", iotaVar)
}
