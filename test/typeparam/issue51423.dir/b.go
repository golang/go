package b

import "./a"

func C() a.Comparator[int] {
	return a.CompareInt[int]
}

func main() {
	_ = C()(1, 2)
}
