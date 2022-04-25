package c

func instantiated[X any](x *X) int {
	if x == nil {
		print(*x) // not reported until _Instances are added to SrcFuncs
	}
	return 1
}

var g int

func init() {
	g = instantiated[int](&g)
}
