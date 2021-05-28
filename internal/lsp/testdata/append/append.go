package append

func foo([]string)  {}
func bar(...string) {}

func _() {
	var (
		aInt     []int    //@item(appendInt, "aInt", "[]int", "var")
		aStrings []string //@item(appendStrings, "aStrings", "[]string", "var")
		aString  string   //@item(appendString, "aString", "string", "var")
	)

	append(aStrings, a)                     //@rank(")", appendString, appendInt)
	var _ interface{} = append(aStrings, a) //@rank(")", appendString, appendInt)
	var _ []string = append(oops, a)        //@rank(")", appendString, appendInt)

	foo(append())                  //@rank("))", appendStrings, appendInt),rank("))", appendStrings, appendString)
	foo(append([]string{}, a))     //@rank("))", appendStrings, appendInt),rank("))", appendString, appendInt),snippet("))", appendStrings, "aStrings...", "aStrings...")
	foo(append([]string{}, "", a)) //@rank("))", appendString, appendInt),rank("))", appendString, appendStrings)

	// Don't add "..." to append() argument.
	bar(append()) //@snippet("))", appendStrings, "aStrings", "aStrings")

	type baz struct{}
	baz{}                       //@item(appendBazLiteral, "baz{}", "", "var")
	var bazzes []baz            //@item(appendBazzes, "bazzes", "[]baz", "var")
	var bazzy baz               //@item(appendBazzy, "bazzy", "baz", "var")
	bazzes = append(bazzes, ba) //@rank(")", appendBazzy, appendBazLiteral, appendBazzes)

	var b struct{ b []baz }
	b.b                  //@item(appendNestedBaz, "b.b", "[]baz", "field")
	b.b = append(b.b, b) //@rank(")", appendBazzy, appendBazLiteral, appendNestedBaz)

	var aStringsPtr *[]string  //@item(appendStringsPtr, "aStringsPtr", "*[]string", "var")
	foo(append([]string{}, a)) //@snippet("))", appendStringsPtr, "*aStringsPtr...", "*aStringsPtr...")

	foo(append([]string{}, *a)) //@snippet("))", appendStringsPtr, "aStringsPtr...", "aStringsPtr...")
}
