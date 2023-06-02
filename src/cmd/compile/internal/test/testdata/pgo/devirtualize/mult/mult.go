package mult

var sink int

type Multiplier interface {
	Multiply(a, b int) int
}

type Mult struct{}

func (Mult) Multiply(a, b int) int {
	for i := 0; i < 1000; i++ {
		sink++
	}
	return a * b
}

type NegMult struct{}

func (NegMult) Multiply(a, b int) int {
	for i := 0; i < 1000; i++ {
		sink++
	}
	return -1 * a * b
}
