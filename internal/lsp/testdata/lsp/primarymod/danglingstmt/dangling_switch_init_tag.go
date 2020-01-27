package danglingstmt

func _() {
	switch i := 0; baz //@rank(" //", danglingBaz2)
}

func baz2() int { //@item(danglingBaz2, "baz2", "func() int", "func")
	return 0
}
