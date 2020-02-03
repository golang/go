package constant

const x = 1 //@item(constX, "x", "int", "const")

const (
	a int = iota << 2 //@item(constA, "a", "int", "const")
	b                 //@item(constB, "b", "int", "const")
	c                 //@item(constC, "c", "int", "const")
)

func _() {
	const y = "hi" //@item(constY, "y", "string", "const")
	//@complete("", constY, constA, constB, constC, constX)
}
