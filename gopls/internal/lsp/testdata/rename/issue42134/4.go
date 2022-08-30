package issue42134

func _() {
	// a is equal to 5. Comment must stay the same

	a := 5
	_ = a //@rename("a", "b")
}
