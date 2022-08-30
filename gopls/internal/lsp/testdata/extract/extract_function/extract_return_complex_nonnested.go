package extract

import "fmt"

func _() (int, string, error) {
	x := 1
	y := "hello"
	z := "bye" //@mark(exSt10, "z")
	if y == z {
		return x, y, fmt.Errorf("same")
	} else {
		z = "hi"
		return x, z, nil
	}
	return x, z, nil //@mark(exEn10, "nil")
	//@extractfunc(exSt10, exEn10)
}
