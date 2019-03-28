package main

var x = 42 //@symbol("x", "x", 13)

const y = 43 //@symbol("y", "y", 14)

type Foo struct { //@symbol("Foo", "Foo", 23)
	Quux
	Bar int
	baz string
}

type Quux struct { //@symbol("Quux", "Quux", 23)
	X float64
}

func (f Foo) Baz() string { //@symbol("Baz", "Baz", 6)
	return f.baz
}

func main() { //@symbol("main", "main", 12)

}

type Stringer interface { //@symbol("Stringer", "Stringer", 11)
	String() string
}
