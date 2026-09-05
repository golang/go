package a

type I interface{ M() int }

type impl struct{ x int }

//go:noinline
func (p *impl) M() int { return p.x }

//go:noinline
func pad() {}

func New(n int) I {
	if n < 0 {
		pad()
		pad()
	}
	return &impl{x: n}
}

type myErr struct{}

func (*myErr) Error() string { return "myErr" }

func Check(n int) error {
	if n < 0 {
		pad()
		pad()
		return &myErr{}
	}
	return nil
}
