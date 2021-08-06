package shadow

func _() {
	a := true
	b, c, _ := A(), B(), D() //@rename("A", "a"),rename("B", "b"),rename("b", "c"),rename("D", "d")
	d := false
	_, _, _, _ = a, b, c, d
}

func A() int {
	return 0
}

func B() int {
	return 0
}

func D() int {
	return 0
}
