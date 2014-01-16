// +build ignore

package main

// Analysis abstraction of recursive calls is finite.

func main() {
	main()
}

// @calls main.main -> main.main
