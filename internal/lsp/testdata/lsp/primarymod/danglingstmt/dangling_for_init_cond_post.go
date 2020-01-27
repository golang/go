package danglingstmt

func _() {
	for i := bar4(); i > bar4(); i += bar //@rank(" //", danglingBar4)
}

func bar4() int { //@item(danglingBar4, "bar4", "func() int", "func")
	return 0
}
