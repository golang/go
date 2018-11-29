package main

import _ "./linkname1"
import "./linkname2"

func main() { // ERROR "can inline main"
	str := "hello/world"
	bs := []byte(str)        // ERROR "\(\[\]byte\)\(str\) escapes to heap"
	if y.ContainsSlash(bs) { // ERROR "inlining call to y.ContainsSlash"
	}
}
