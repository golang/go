package inlayHint //@inlayHint("package")

const True = true

type Kind int

const (
	KindNone Kind = iota
	KindPrint
	KindPrintf
	KindErrorf
)

const (
	u         = iota * 4
	v float64 = iota * 42
	w         = iota * 42
)

const (
	a, b = 1, 2
	c, d
	e, f = 5 * 5, "hello" + "world"
	g, h
	i, j = true, f
)

// No hint
const (
	Int     = 3
	Float   = 3.14
	Bool    = true
	Rune    = '3'
	Complex = 2.7i
	String  = "Hello, world!"
)

var (
	varInt     = 3
	varFloat   = 3.14
	varBool    = true
	varRune    = '3' + '4'
	varComplex = 2.7i
	varString  = "Hello, world!"
)
