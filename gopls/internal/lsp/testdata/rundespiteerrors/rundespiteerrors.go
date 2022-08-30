package rundespiteerrors

// This test verifies that analyzers without RunDespiteErrors are not
// executed on a package containing type errors (see issue #54762).
func _() {
	// A type error.
	_ = 1 + "" //@diag("1", "compiler", "mismatched types|cannot convert", "error")

	// A violation of an analyzer for which RunDespiteErrors=false:
	// no diagnostic is produced; the diag comment is merely illustrative.
	for _ = range "" { //diag("for _", "simplifyrange", "simplify range expression", "warning")

	}
}
