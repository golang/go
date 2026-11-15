// run

package main

import (
	"fmt"
	"os"
)

var failed bool

func check(name string, cond bool, got, want any) {
	if !cond {
		failed = true
		fmt.Println("FAIL", name+":", "got", got, "want", want)
		return
	}
	fmt.Println("ok", name)
}

func main() {
	s := "twin-ok"
	check("twin", s == "twin-ok", s, "twin-ok")
	if failed {
		os.Exit(1)
	}
}
