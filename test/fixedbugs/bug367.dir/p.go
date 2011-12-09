package p

type T struct{ x int }
type S struct{}

func (p *S) get() {
}

type I interface {
	get()
}

func F(i I) {
	i.get()
}
