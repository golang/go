package danglingstmt

func _() {
	switch i := baz //@rank(" //", danglingBaz)
}

func baz() int { //@item(danglingBaz, "baz", "func() int", "func")
	return 0
}
