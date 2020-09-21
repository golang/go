package extract

func _() {
	var a []int
	a = append(a, 2) //@mark(exSt6, "a")
	b := 4           //@mark(exEn6, "4")
	//@extractfunc(exSt6, exEn6)
	a = append(a, b)
}
