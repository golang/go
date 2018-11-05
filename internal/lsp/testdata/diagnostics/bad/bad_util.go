// +build go1.11

package bad

func random2(y int) int {
	x := 6 //@diag("x", "x declared but not used")
	return y
}
