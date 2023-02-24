package a

type Atyp int

func (ap *Atyp) Set(q int) {
	*ap = Atyp(q)
}

func (ap Atyp) Get() int {
	inter := func(q Atyp) int {
		return int(q)
	}
	return inter(ap)
}

var afunc = func(x int) int {
	return x + 1
}
var Avar = afunc(42)

func A(x int) int {
	if x == 0 {
		return 22
	} else if x == 1 {
		return 33
	}
	return 44
}
