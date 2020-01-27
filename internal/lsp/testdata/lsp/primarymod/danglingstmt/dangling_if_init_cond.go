package danglingstmt

func _() {
	if i := 123; foo //@rank(" //", danglingFoo3)
}

func foo3() bool { //@item(danglingFoo3, "foo3", "func() bool", "func")
	return true
}
