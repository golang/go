package missingfunction

func errorParam() {
	var err error
	undefinedErrorParam(err) //@suggestedfix("undefinedErrorParam", "quickfix")
}
