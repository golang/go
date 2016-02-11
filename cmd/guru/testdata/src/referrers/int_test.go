package main

import "lib"

func _() {
	// This reference should be found by the ref-method query.
	_ = (lib.Type).Method // ref from internal test package
}
