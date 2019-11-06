package append

func foo([]string)  {}
func bar(...string) {}

func _() {
	var (
		aInt     []int    //@item(appendInt, "aInt", "[]int", "var")
		aStrings []string //@item(appendStrings, "aStrings", "[]string", "var")
		aString  string   //@item(appendString, "aString", "string", "var")
	)

	foo(append())           //@rank("))", appendStrings, appendInt),rank("))", appendStrings, appendString)
	foo(append(nil, a))     //@rank("))", appendStrings, appendInt),rank("))", appendString, appendInt),snippet("))", appendStrings, "aStrings...", "aStrings...")
	foo(append(nil, "", a)) //@rank("))", appendString, appendInt),rank("))", appendString, appendStrings)

	// Don't add "..." to append() argument.
	bar(append()) //@snippet("))", appendStrings, "aStrings", "aStrings")
}
