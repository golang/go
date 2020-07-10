package extract

func _() {
	x0 := 1
	a := x0 //@extractfunc("a", "x0")
}

func x1() int {
	return 1
}
