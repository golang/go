package extract

func _() {
	a := /* comment in the middle of a line */ 1 //@mark(exSt18, "a")
	// Comment on its own line  //@mark(exSt19, "Comment")
	_ = 3 + 4 //@mark(exEn18, "4"),mark(exEn19, "4"),mark(exSt20, "_")
	// Comment right after 3 + 4

	// Comment after with space //@mark(exEn20, "Comment")

	//@extractfunc(exSt18, exEn18),extractfunc(exSt19, exEn19),extractfunc(exSt20, exEn20)
}
