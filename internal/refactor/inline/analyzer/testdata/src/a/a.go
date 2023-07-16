package a

func f() {
	One()        // want `inline call of a.One`
	new(T).Two() // want `inline call of \(a.T\).Two`
}

type T struct{}

// inlineme
func One() int { return one } // want One:`inlineme a.One`

const one = 1

// inlineme
func (T) Two() int { return 2 } // want Two:`inlineme \(a.T\).Two`
