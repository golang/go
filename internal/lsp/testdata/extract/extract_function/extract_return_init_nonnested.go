package extract

func _() string {
	x := 1
	if x == 0 { //@mark(exSt12, "if")
		x = 3
		return "a"
	}
	x = 2
	return "b" //@mark(exEn12, "\"b\"")
	//@extractfunc(exSt12, exEn12)
}
