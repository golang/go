package D

type T0 int

export type T1 struct {
	n int;
	a, b T0;
}

export type T2 struct {
	u, v float;
}

export func (obj *T2) M1(u, v float) {
}
