package main_test

import "lib"

func _() {
	// This reference should be found by the ref-method query.
	_ = (lib.Type).Method // ref from external test package
}
