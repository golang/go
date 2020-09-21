package rank

func _() {
	someRandomBoolFunc := func() bool { //@item(boolExprFunc, "someRandomBoolFunc", "func() bool", "var")
		return true
	}

	var foo, bar int     //@item(boolExprBar, "bar", "int", "var")
	if foo == 123 && b { //@rank(" {", boolExprBar, boolExprFunc)
	}
}
