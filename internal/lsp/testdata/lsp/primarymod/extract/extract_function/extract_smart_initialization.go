package extract

func _() {
	var a []int
	a = append(a, 2) //@mark(s4, "a")
	b := 4           //@mark(e4, "4")
	//@extractfunc(s4, e4)
	a = append(a, b)
}
