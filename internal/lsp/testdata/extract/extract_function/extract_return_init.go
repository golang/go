package extract

func _() string {
	x := 1
	if x == 0 { //@mark(exSt5, "if")
		x = 3
		return "a"
	} //@mark(exEn5, "}")
	x = 2
	return "b"
	//@extractfunc(exSt5, exEn5)
}
