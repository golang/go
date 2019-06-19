package funcvalue

func fooFunc() int { //@item(fvFooFunc, "fooFunc", "func() int", "func")
	return 0
}

var _ = fooFunc() //@item(fvFooFuncCall, "fooFunc()", "int", "func")

var fooVar = func() int { //@item(fvFooVar, "fooVar", "func() int", "var")
	return 0
}

var _ = fooVar() //@item(fvFooVarCall, "fooVar()", "int", "var")

type myFunc func() int

var fooType myFunc = fooVar //@item(fvFooType, "fooType", "myFunc", "var")

var _ = fooType() //@item(fvFooTypeCall, "fooType()", "int", "var")

func _() {
	var f func() int
	f = foo //@complete(" //", fvFooFunc, fvFooType, fvFooVar)

	var i int
	i = foo //@complete(" //", fvFooFuncCall, fvFooTypeCall, fvFooVarCall)
}
