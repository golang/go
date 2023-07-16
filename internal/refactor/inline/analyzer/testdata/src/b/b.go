package b

import "a"

func f() {
	a.One() // want `cannot inline call to a.One because body refers to non-exported one`

	new(a.T).Two() // want `inline call of \(a.T\).Two`
}
