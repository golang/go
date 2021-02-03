package index

func _() {
	var (
		aa = "123" //@item(indexAA, "aa", "string", "var")
		ab = 123   //@item(indexAB, "ab", "int", "var")
	)

	var foo [1]int
	foo[a]  //@complete("]", indexAB, indexAA)
	foo[:a] //@complete("]", indexAB, indexAA)
	a[:a]   //@complete("[", indexAA, indexAB)
	a[a]    //@complete("[", indexAA, indexAB)

	var bar map[string]int
	bar[a] //@complete("]", indexAA, indexAB)

	type myMap map[string]int
	var baz myMap
	baz[a] //@complete("]", indexAA, indexAB)

	type myInt int
	var mi myInt //@item(indexMyInt, "mi", "myInt", "var")
	foo[m]       //@snippet("]", indexMyInt, "mi", "mi")
}
