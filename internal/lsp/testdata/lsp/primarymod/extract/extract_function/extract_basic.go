package extract

func _() {
	a := 1        //@mark(s0, "a")
	var _ = 3 + 4 //@mark(e0, "4")
	//@extractfunc(s0, e0)
}
