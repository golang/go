package D

type T0 int

type T1 struct {
	n int;
	a, b T0;
}

type T2 struct {
	u, v float;
}

func (obj *T2) M1(u, v float) {
}

func F0(a int, b T0) int {
	return a + b;
}