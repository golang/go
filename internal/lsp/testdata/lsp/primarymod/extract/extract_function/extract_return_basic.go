package extract

func _() bool {
	x := 1
	if x == 0 { //@mark(s0, "if")
		return true
	} //@mark(e0, "}")
	return false
	//@extractfunc(s0, e0)
}
