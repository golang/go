package main

// Need to compile package gob with debug.go to build this program.

import (
	"encoding/gob"
	"fmt"
	"os"
)

func main() {
	var err error
	file := os.Stdin
	if len(os.Args) > 1 {
		file, err = os.Open(os.Args[1])
		if err != nil {
			fmt.Fprintf(os.Stderr, "dump: %s\n", err)
			os.Exit(1)
		}
	}
	gob.Debug(file)
}
