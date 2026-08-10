package a

type I interface{ M() int }

type impl struct{ x int }

//go:noinline
func (p *impl) M() int { return p.x } // ERROR "p does not escape"

//go:noinline
func pad() {}

func New(n int) I { // ERROR "result #0 of New is always \*impl" "splitting New into New\.dv"
	if n < 0 {
		pad()
		pad()
	}
	return &impl{x: n} // ERROR "&impl{...} escapes to heap"
}

type myErr struct{}

//go:noinline
func (*myErr) Error() string { return "myErr" }

// Exported as {*myErr, nil}; not split, but callers can reason about
// the receiver.
func Check(n int) error { // ERROR "result #0 of Check is one of \*myErr, <nil>"
	if n < 0 {
		pad()
		pad()
		return &myErr{} // ERROR "&myErr{} escapes to heap"
	}
	return nil
}
