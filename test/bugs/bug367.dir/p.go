package p

type T struct{ x int }
type S struct{}

func (p *S) get() T {
	return T{0}
}

type I interface {
	get() T
}

func F(i I) {
	_ = i.get()
}
