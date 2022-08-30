package extract

func _() {
	a := /* comment in the middle of a line */ 1 //@mark(exSt18, "a")
	// Comment on its own line
	_ = 3 + 4 //@mark(exEn18, "4")
	//@extractfunc(exSt18, exEn18)
}
