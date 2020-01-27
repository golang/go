package danglingstmt

func _() {
	for i := bar3(); i > bar //@rank(" //", danglingBar3)
}

func bar3() int { //@item(danglingBar3, "bar3", "func() int", "func")
	return 0
}
