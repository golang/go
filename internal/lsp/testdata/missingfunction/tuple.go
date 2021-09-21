package missingfunction

func tuple() {
	undefinedTuple(b()) //@suggestedfix("undefinedTuple", "quickfix")
}

func b() (string, error) {
	return "", nil
}
