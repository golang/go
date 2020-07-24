package extract

func _() {
	var b []int
	var a int
	a = 2 //@mark(exSt7, "a")
	b = []int{}
	b = append(b, a) //@mark(exEn7, ")")
	b[0] = 1
	//@extractfunc(exSt7, exEn7)
}
