package main

import (
	"bad_select_crash.dir/genCaller42"
	"bad_select_crash.dir/genUtils"
	"fmt"
	"os"
)

func main() {
	// Only print if there is a problem
	genCaller42.Caller2()
	if genUtils.FailCount != 0 {
		fmt.Fprintf(os.Stderr, "FAILURES: %d\n", genUtils.FailCount)
		os.Exit(2)
	}
}
