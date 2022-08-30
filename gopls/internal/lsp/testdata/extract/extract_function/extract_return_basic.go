package extract

func _() bool {
	x := 1
	if x == 0 { //@mark(exSt2, "if")
		return true
	} //@mark(exEn2, "}")
	return false
	//@extractfunc(exSt2, exEn2)
}
