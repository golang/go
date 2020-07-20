package extract

import "fmt"

func _() (int, string, error) {
	x := 1
	y := "hello"
	z := "bye" //@mark(s0, "z")
	if y == z {
		return x, y, fmt.Errorf("same")
	} else {
		z = "hi"
		return x, z, nil
	} //@mark(e0, "}")
	return x, z, nil
	//@extractfunc(s0, e0)
}
