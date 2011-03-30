package p

func _() {
	// the scope of a local type declaration starts immediately after the type name
	type T struct{ _ *T }
}
