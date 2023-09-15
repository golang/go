package danglingstmt

// TODO: re-enable this test, which was broken when the foo package was removed.
// (we can replicate the relevant definitions in the new marker test)
// import "golang.org/lsptests/foo"

func _() {
	foo. // rank(" //", Foo)
	var _ = []string{foo.} // rank("}", Foo)
}
