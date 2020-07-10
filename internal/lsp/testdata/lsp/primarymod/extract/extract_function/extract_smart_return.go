package extract

func _() {
	var b []int
	var a int
	a = 2 //@mark(s2, "a")
	b = []int{}
	b = append(b, a) //@mark(e2, ")")
	b[0] = 1
	//@extractfunc(s2, e2)
}
