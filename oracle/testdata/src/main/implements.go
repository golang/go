package main

// Tests of 'implements' query.
// See go.tools/oracle/oracle_test.go for explanation.
// See implements.golden for expected query results.

// @implements impl ""

func main() {
}

type E interface{}

type F interface {
	f()
}

type FG interface {
	f()
	g() int
}

type C int
type D struct{}

func (c *C) f() {}
func (d D) f()  {}

func (d *D) g() int { return 0 }
