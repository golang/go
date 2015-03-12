package main

// Tests of 'implements' query applied to methods.
// See go.tools/oracle/oracle_test.go for explanation.
// See implements-methods.golden for expected query results.

import _ "lib"

func main() {
}

type F interface {
	f() // @implements F.f "f"
}

type FG interface {
	f()       // @implements FG.f "f"
	g() []int // @implements FG.g "g"
}

type C int
type D struct{}

func (c *C) f() {} // @implements *C.f "f"
func (d D) f()  {} // @implements D.f "f"

func (d *D) g() []int { return nil } // @implements *D.g "g"

type sorter []int

func (sorter) Len() int           { return 0 } // @implements Len "Len"
func (sorter) Less(i, j int) bool { return false }
func (sorter) Swap(i, j int)      {}

type I interface {
	Method(*int) *int // @implements I.Method "Method"
}
