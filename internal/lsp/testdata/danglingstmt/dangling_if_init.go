package danglingstmt

func _() {
	if i := foo //@rank(" //", danglingFoo2)
}

func foo2() bool { //@item(danglingFoo2, "foo2", "func() bool", "func")
	return true
}
