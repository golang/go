// errorcheck -0 -m

package foo

func alloc(x int) *int { // ERROR "can inline alloc" "moved to heap: x"
	return &x
}
