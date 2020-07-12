package another

type (
	I interface{ F() }
	C struct{ I }
)

func (C) g()

func _() {
	var x I = C{}
	x.F() //@rename("F", "G")
}
