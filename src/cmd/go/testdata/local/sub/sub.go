package sub

import (
	"fmt"

	subsub "./sub"
)

func Hello() {
	fmt.Println("sub.Hello")
	subsub.Hello()
}
