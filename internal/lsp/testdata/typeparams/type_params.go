//go:build go1.18
// +build go1.18

package typeparams

func one[a int | string]()            {}
func two[a int | string, b float64 | int]() {}

func _() {
	one[]() //@rank("]", string, float64)
	two[]() //@rank("]", int, float64)
	two[int, f]() //@rank("]", float64, float32)
}
