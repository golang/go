package format //@format("package")

import (
	"fmt"
	"runtime"

	"log"
)

func hello() {

	var x int //@diag("x", "x declared but not used")
}

func hi() {

	runtime.GOROOT()
	fmt.Printf("")

	log.Printf("")
}
