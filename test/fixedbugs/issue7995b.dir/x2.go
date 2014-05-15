package main

import "./x1"

func main() {
	s := x1.F(&x1.P)
	if s != "100 100\n" {
		println("BUG:", s)
	}
}
