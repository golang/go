package main

// Tests of 'implements' query, -output=json.
// See go.tools/guru/guru_test.go for explanation.
// See implements.golden for expected query results.

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
