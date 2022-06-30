package missingfunction

func slice() {
	undefinedSlice([]int{1, 2}) //@suggestedfix("undefinedSlice", "quickfix", "")
}
