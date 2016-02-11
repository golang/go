package describe // @describe pkgdecl "describe"

// Tests of 'describe' query, -format=json.
// See go.tools/guru/guru_test.go for explanation.
// See describe-json.golden for expected query results.

func main() {
	var s struct{ x [3]int }
	p := &s.x[0] // @describe desc-val-p "p"
	_ = p

	var i I = C(0)
	if i == nil {
		i = new(D)
	}
	print(i) // @describe desc-val-i "\\bi\\b"

	go main() // @describe desc-stmt "go"
}

type I interface {
	f()
}

type C int // @describe desc-type-C "C"
type D struct{}

func (c C) f()  {}
func (d *D) f() {}
