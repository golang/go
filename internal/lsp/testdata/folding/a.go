package folding //@fold("package")

import (
	"fmt"
	_ "log"
)

import _ "os"

// bar is a function.
// With a multiline doc comment.
func bar() string {
	switch {
	case true:
		if true {
			fmt.Println("true")
		} else {
			fmt.Println("false")
		}
	case false:
		fmt.Println("false")
	default:
		fmt.Println("default")
	}
	_ = []int{
		1,
		2,
		3,
	}
	_ = [2]string{"d",
		"e"
	}
	_ = map[string]int{
		"a": 1,
		"b": 2,
		"c": 3,
	}
	type T struct {
		f string
		g int
		h string
	}
	_ = T{
		f: "j",
		g: 4,
		h: "i",
	}
	x, y := make(chan bool), make(chan bool)
	select {
	case val := <-x:
		if val {
			fmt.Println("true from x")
		} else {
			fmt.Println("false from x")
		}
	case <-y:
		fmt.Println("y")
	default:
		fmt.Println("default")
	}
	// This is a multiline comment
	// that is not a doc comment.
	return `
this string
is not indented`
}
