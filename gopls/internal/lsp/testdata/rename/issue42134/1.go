package issue42134

func _() {
	// foo computes things.
	foo := func() {}

	foo() //@rename("foo", "bar")
}
