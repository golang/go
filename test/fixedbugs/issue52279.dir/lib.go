package lib

type FMap[K comparable, V comparable] map[K]V

//go:noinline
func (m FMap[K, V]) Flip() FMap[V, K] {
	out := make(FMap[V, K])
	return out
}

type MyType uint8

const (
	FIRST MyType = 0
)

var typeStrs = FMap[MyType, string]{
	FIRST: "FIRST",
}

func (self MyType) String() string {
	return typeStrs[self]
}
