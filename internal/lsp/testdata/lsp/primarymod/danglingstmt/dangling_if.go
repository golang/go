package danglingstmt

func _() {
	if foo //@rank(" //", danglingFoo)
}

func foo() bool { //@item(danglingFoo, "foo", "func() bool", "func")
	return true
}
