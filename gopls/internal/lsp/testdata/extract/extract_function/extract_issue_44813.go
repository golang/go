package extract

import "fmt"

func main() {
	x := []rune{} //@mark(exSt9, "x")
	s := "HELLO"
	for _, c := range s {
		x = append(x, c)
	} //@mark(exEn9, "}")
	//@extractfunc(exSt9, exEn9)
	fmt.Printf("%x\n", x)
}
