// compile

package main

type Foo interface {
	Print()
}

type Bar struct{}

func (b Bar) Print() {}

func main() {
	b := make([]Bar, 20)
	f := make([]Foo, 20)
	for i := range f {
		f[i] = b[i]
	}
	T(f)
	_ = make([]struct{}, 1)
}

func T(f []Foo) {
	for i := range f {
		f[i].Print()
	}
}
