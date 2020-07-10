package extract

func _() {
	a := 1
	a = 5     //@mark(s1, "a")
	a = a + 2 //@mark(e1, "2")
	//@extractfunc(s1, e1)
	b := a * 2
	var _ = 3 + 4
}
