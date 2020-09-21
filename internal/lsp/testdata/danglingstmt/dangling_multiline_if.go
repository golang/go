package danglingstmt

func walrus() bool { //@item(danglingWalrus, "walrus", "func() bool", "func")
	return true
}

func _() {
	if true &&
		walrus //@complete(" //", danglingWalrus)
}
