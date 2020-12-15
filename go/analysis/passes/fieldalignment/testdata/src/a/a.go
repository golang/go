package a

type Good struct {
	y int32
	x byte
	z byte
}

type Bad struct { // want "struct of size 12 could be 8"
	x byte
	y int32
	z byte
}

type ZeroGood struct {
	a [0]byte
	b uint32
}

type ZeroBad struct { // want "struct of size 8 could be 4"
	a uint32
	b [0]byte
}

type NoNameGood struct {
	Good
	y int32
	x byte
	z byte
}

type NoNameBad struct { // want "struct of size 20 could be 16"
	Good
	x byte
	y int32
	z byte
}
