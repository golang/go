package b

var q int

func Top(x int) int {
	q += 1
	if q != x {
		return 3
	}
	return 4
}

func OOO(x int) int {
	defer func() { q += x & 7 }()
	return Top(x + 1)
}
