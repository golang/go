package danglingstmt

func _() {
	for bar //@rank(" //", danglingBar)
}

func bar() bool { //@item(danglingBar, "bar", "func() bool", "func")
	return true
}
