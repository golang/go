package extract

func _() string {
	x := 1
	if x == 0 { //@mark(s0, "if")
		x = 3
		return "a"
	} //@mark(e0, "}")
	x = 2
	return "b"
	//@extractfunc(s0, e0)
}
