package main

import "dep2"

func main() {
	dep2.W = dep2.G() + 1
}
