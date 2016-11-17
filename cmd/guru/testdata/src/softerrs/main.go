package main

// Tests of various queries on a program containing only "soft" errors.
// See go.tools/guru/guru_test.go for explanation.
// See main.golden for expected query results.

func _() {
	var i int // "unused var" is a soft error
}

func f() {} // @callers softerrs-callers-f "f"

func main() {
	f() // @describe softerrs-describe-f "f"
}
