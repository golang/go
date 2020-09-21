package extract

func _() {
	fn0 := 1
	a := fn0 //@extractfunc("a", "fn0")
}

func fn1() int {
	return 1
}
