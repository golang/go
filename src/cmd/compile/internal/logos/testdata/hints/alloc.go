package p

type T struct{ A, B int }

var sink *T

func keep() { sink = &T{1, 2} }

func local() int {
	t := &T{3, 4}
	return t.A + t.B
}

func grow(n int) []byte {
	return make([]byte, n)
}

func moved() *int {
	x := 5
	return &x
}
