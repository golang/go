package extract

func _() {
	newFunction := 1
	a := newFunction //@extractfunc("a", "newFunction")
}

func newFunction1() int {
	return 1
}
