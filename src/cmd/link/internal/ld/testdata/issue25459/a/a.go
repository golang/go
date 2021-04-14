package a

const Always = true

var Count int

type FuncReturningInt func() int

var PointerToConstIf FuncReturningInt

func ConstIf() int {
	if Always {
		return 1
	}
	var imdead [4]int
	imdead[Count] = 1
	return imdead[0]
}

func CallConstIf() int {
	Count += 3
	return ConstIf()
}

func Another() {
	defer func() { PointerToConstIf = ConstIf; Count += 1 }()
}
