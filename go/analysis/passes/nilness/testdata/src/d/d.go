package d

type message interface{ PR() }

func noparam() {
	var messageT message
	messageT.PR() // want "nil dereference in dynamic method call"
}

func paramNonnil[T message]() {
	var messageT T
	messageT.PR() // cannot conclude messageT is nil.
}

func instance() {
	// buildssa.BuilderMode does not include InstantiateGenerics.
	paramNonnil[message]() // no warning is expected as param[message] id not built.
}

func param[T interface {
	message
	~*int | ~chan int
}]() {
	var messageT T // messageT is nil.
	messageT.PR()  // nil receiver may be okay. See param[nilMsg].
}

type nilMsg chan int

func (m nilMsg) PR() {
	if m == nil {
		print("not an error")
	}
}

var G func() = param[nilMsg] // no warning

func allNillable[T ~*int | ~chan int]() {
	var x, y T  // both are nillable and are nil.
	if x != y { // want "impossible condition: nil != nil"
		print("unreachable")
	}
}

func notAll[T ~*int | ~chan int | ~int]() {
	var x, y T  // neither are nillable due to ~int
	if x != y { // no warning
		print("unreachable")
	}
}

func noninvoke[T ~func()]() {
	var x T
	x() // want "nil dereference in dynamic function call"
}
