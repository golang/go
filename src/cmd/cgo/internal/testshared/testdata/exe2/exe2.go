package main

import "testshared/dep2"

func main() {
	d := &dep2.Dep2{}
	dep2.W = dep2.G() + 1 + d.Method()
}
