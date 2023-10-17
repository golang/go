package aliases

type (
	T0 [10]int
	T1 []byte
	T2 struct {
		x int
	}
	T3 interface {
		m() T2
	}
	T4 func(int, T0) chan T2
)

// basic aliases
type (
	Ai = int
	A0 = T0
	A1 = T1
	A2 = T2
	A3 = T3
	A4 = T4

	A10 = [10]int
	A11 = []byte
	A12 = struct {
		x int
	}
	A13 = interface {
		m() A2
	}
	A14 = func(int, A0) chan A2
)

// alias receiver types
func (T0) m1() {}
func (A0) m2() {}

// alias receiver types (long type declaration chains)
type (
	V0 = V1
	V1 = (V2)
	V2 = (V3)
	V3 = T0
)

func (V1) n() {}

// cycles
type C0 struct {
	f1 C1
	f2 C2
}

type (
	C1 *C0
	C2 = C1
)

type (
	C5 struct {
		f *C6
	}
	C6 = C5
)
