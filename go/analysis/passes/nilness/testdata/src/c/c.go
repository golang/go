package c

func instantiated[X any](x *X) int {
	if x == nil {
		print(*x) // want "nil dereference in load"
	}
	return 1
}

var g int

func init() {
	g = instantiated[int](&g)
}
