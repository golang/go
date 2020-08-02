package builtins

func _() {
	var _ []bool //@item(builtinBoolSliceType, "[]bool", "[]bool", "type")

	var _ []bool = make() //@rank(")", builtinBoolSliceType, int)

	var _ []bool = make([], 0) //@rank(",", bool, int)

	var _ [][]bool = make([][], 0) //@rank(",", bool, int)
}
