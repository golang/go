// +build ignore

package D1

import "fmt"

func example() {
	fmt.Println(123, "a")         // match
	fmt.Println(0x7b, `a`)        // match
	fmt.Println(0173, "\x61")     // match
	fmt.Println(100+20+3, "a"+"") // no match: constant expressions, but not basic literals
}
