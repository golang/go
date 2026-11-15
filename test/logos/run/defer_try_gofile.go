// run

package main

import (
	"fmt"
	"os"

	rt "logos/rt"
)

var order []string

func f() error {
	order = append(order, "f")
	return nil
}

func guarded() (err error) {
	defer rt.DeferTry(&err, f())
	order = append(order, "after-defer-statement")
	return nil
}

func main() {
	if err := guarded(); err != nil {
		fmt.Println("FAIL gofile")
		os.Exit(1)
	}
	if len(order) == 2 && order[0] == "f" && order[1] == "after-defer-statement" {
		fmt.Println("ok gofile")
	} else {
		fmt.Println("FAIL gofile")
		os.Exit(1)
	}
}
