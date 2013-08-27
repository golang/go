package lib

type Type int

func (Type) Method(x *int) *int {
	return x
}

func Func() {
}

const Const = 3

var Var = 0
