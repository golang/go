package p2

type Twoer interface {
	PackageTwoMeth()
}

func F() string               {}
func G() Twoer                {}
func NewError(s string) error {}
