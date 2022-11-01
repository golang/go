package issues

// Test for golang/go#56505: completion on variables of type *error should not
// panic.
func _() {
	var e *error
	e.x //@complete(" //")
}
