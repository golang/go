//gofmt -s

// Test cases for range simplification.
package p

func _() {
	for a, b = range x {}
	for a, _ = range x {}
	for _, b = range x {}
	for _, _ = range x {}

	for a = range x {}
	for _ = range x {}

	for a, b := range x {}
	for a, _ := range x {}
	for _, b := range x {}

	for a := range x {}
}
