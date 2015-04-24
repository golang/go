package main_test

import (
	"lib"
	renamed "referrers" // package has name "main", path "referrers", local name "renamed"
)

func _() {
	// This reference should be found by the ref-method query.
	_ = (lib.Type).Method // ref from external test package
	var _ renamed.T
}
