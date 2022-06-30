package missingfunction

func uniqueArguments() {
	var s string
	var i int
	undefinedUniqueArguments(s, i, s) //@suggestedfix("undefinedUniqueArguments", "quickfix", "")
}
