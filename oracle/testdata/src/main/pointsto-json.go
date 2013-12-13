package pointsto

// Tests of 'pointsto' queries, -format=json.
// See go.tools/oracle/oracle_test.go for explanation.
// See pointsto-json.golden for expected query results.

func main() { //
	var s struct{ x [3]int }
	p := &s.x[0] // @pointsto val-p "p"
	_ = p

	var i I = C(0)
	if i == nil {
		i = new(D)
	}
	print(i) // @pointsto val-i "\\bi\\b"
}

type I interface {
	f()
}

type C int
type D struct{}

func (c C) f()  {}
func (d *D) f() {}
