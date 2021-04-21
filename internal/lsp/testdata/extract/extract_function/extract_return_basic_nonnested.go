package extract

func _() bool {
	x := 1 //@mark(exSt13, "x")
	if x == 0 {
		return true
	}
	return false //@mark(exEn13, "false")
	//@extractfunc(exSt13, exEn13)
}
