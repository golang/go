//go:build ignore
// +build ignore

package main

// Analysis abstraction of recursive calls is finite.

func main() {
	main()
}

// @calls command-line-arguments.main -> command-line-arguments.main
