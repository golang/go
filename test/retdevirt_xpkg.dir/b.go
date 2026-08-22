package b

import "./a"

//go:noinline
func Use() int {
	v := a.New(1) // ERROR "devirtualizing call to a\.New\.dv"
	return v.M()  // ERROR "devirtualizing v.M to \*a\.impl"
}
