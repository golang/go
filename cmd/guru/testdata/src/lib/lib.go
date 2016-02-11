package lib

type Type int

func (Type) Method(x *int) *int {
	return x
}

func Func() {
}

const Const = 3

var Var = 0

type Sorter interface {
	Len() int
	Less(i, j int) bool
	Swap(i, j int)
}
