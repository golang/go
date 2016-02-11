package main

// Tests of 'implements' query.
// See go.tools/guru/guru_test.go for explanation.
// See implements.golden for expected query results.

import _ "lib"

func main() {
}

type E interface{} // @implements E "E"

type F interface { // @implements F "F"
	f()
}

type FG interface { // @implements FG "FG"
	f()
	g() []int // @implements slice "..int"
}

type C int // @implements C "C"
type D struct{}

func (c *C) f() {} // @implements starC ".C"
func (d D) f()  {} // @implements D "D"

func (d *D) g() []int { return nil } // @implements starD ".D"

type sorter []int // @implements sorter "sorter"

func (sorter) Len() int           { return 0 }
func (sorter) Less(i, j int) bool { return false }
func (sorter) Swap(i, j int)      {}

type I interface { // @implements I "I"
	Method(*int) *int
}
