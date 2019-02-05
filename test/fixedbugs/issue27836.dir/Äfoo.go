package Äfoo

var ÄbarV int = 101

func Äbar(x int) int {
	defer func() { ÄbarV += 3 }()
	return Äblix(x)
}

func Äblix(x int) int {
	defer func() { ÄbarV += 9 }()
	return ÄbarV + x
}
