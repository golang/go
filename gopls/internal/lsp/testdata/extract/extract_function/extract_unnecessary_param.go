package extract

func _() {
	var b []int
	var a int
	a := 2 //@mark(exSt8, "a")
	b = []int{}
	b = append(b, a) //@mark(exEn8, ")")
	b[0] = 1
	if a == 2 {
		return
	}
	//@extractfunc(exSt8, exEn8)
}
