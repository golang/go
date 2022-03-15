//go:build go1.18
// +build go1.18

package generics

type G[P any] struct {
	F int
}

func (G[_]) M() {}

func F[P any](P) {
	var p P //@rename("P", "Q")
	_ = p
}

func _() {
	var x G[int] //@rename("G", "H")
	_ = x.F      //@rename("F", "K")
	x.M()        //@rename("M", "N")

	var y G[string]
	_ = y.F
	y.M()
}
