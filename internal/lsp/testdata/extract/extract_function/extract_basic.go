package extract

func _() { //@mark(exSt25, "{")
	a := 1    //@mark(exSt1, "a")
	_ = 3 + 4 //@mark(exEn1, "4")
	//@extractfunc(exSt1, exEn1)
	//@extractfunc(exSt25, exEn25)
} //@mark(exEn25, "}")
