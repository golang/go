package typemods

func fooFunc() func() int { //@item(modFooFunc, "fooFunc", "func() func() int", "func")
	return func() int {
		return 0
	}
}

func fooPtr() *int { //@item(modFooPtr, "fooPtr", "func() *int", "func")
	return nil
}

func _() {
	var _ int = foo //@snippet(" //", modFooFunc, "fooFunc()()", "fooFunc()()"),snippet(" //", modFooPtr, "*fooPtr()", "*fooPtr()")
}

func _() {
	var m map[int][]chan int //@item(modMapChanPtr, "m", "map[int]chan *int", "var")

	var _ int = m //@snippet(" //", modMapChanPtr, "<-m[${1:}][${2:}]", "<-m[${1:}][${2:}]")
}
