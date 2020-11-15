package a

type ZeroGood struct {
	a [0]byte
	b uint32
}

type ZeroBad struct { // want "struct of size 8 could be 4"
	a uint32
	b [0]byte
}
