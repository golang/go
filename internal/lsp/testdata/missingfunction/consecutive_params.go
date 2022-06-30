package missingfunction

func consecutiveParams() {
	var s string
	undefinedConsecutiveParams(s, s) //@suggestedfix("undefinedConsecutiveParams", "quickfix", "")
}
