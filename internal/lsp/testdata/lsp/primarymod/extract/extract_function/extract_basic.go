package extract

func _() {
	a := 1    //@mark(exSt1, "a")
	_ = 3 + 4 //@mark(exEn1, "4")
	//@extractfunc(exSt1, exEn1)
}
