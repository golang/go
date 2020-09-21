package danglingstmt

func _() {
	for i := bar //@rank(" //", danglingBar2)
}

func bar2() int { //@item(danglingBar2, "bar2", "func() int", "func")
	return 0
}
