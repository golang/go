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
