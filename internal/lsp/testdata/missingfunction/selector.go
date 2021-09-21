package missingfunction

func selector() {
	m := map[int]bool{}
	undefinedSelector(m[1]) //@suggestedfix("undefinedSelector", "quickfix")
}
