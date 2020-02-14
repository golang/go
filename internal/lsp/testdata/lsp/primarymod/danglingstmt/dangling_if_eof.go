package danglingstmt

func bar5() bool { //@item(danglingBar5, "bar5", "func() bool", "func")
	return true
}

func _() {
	if b //@rank(" //", danglingBar5)
