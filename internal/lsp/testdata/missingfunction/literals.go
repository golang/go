package missingfunction

type T struct{}

func literals() {
	undefinedLiterals("hey compiler", T{}, &T{}) //@suggestedfix("undefinedLiterals", "quickfix", "")
}
