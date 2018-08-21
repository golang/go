// +build !foo-bar

package p1

import "fmt"

func F() {
	fmt.Printf("%d", "hello") // causes vet error
}
