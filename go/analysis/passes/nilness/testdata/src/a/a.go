package a

type X struct{ f, g int }

func f(x, y *X) {
	if x == nil {
		print(x.f) // want "nil dereference in field selection"
	} else {
		print(x.f)
	}

	if x == nil {
		if nil != y {
			print(1)
			panic(0)
		}
		x.f = 1 // want "nil dereference in field selection"
		y.f = 1 // want "nil dereference in field selection"
	}

	var f func()
	if f == nil { // want "tautological condition: nil == nil"
		go f() // want "nil dereference in dynamic function call"
	} else {
		// This block is unreachable,
		// so we don't report an error for the
		// nil dereference in the call.
		defer f()
	}
}

func f2(ptr *[3]int, i interface{}) {
	if ptr != nil {
		print(ptr[:])
		*ptr = [3]int{}
		print(*ptr)
	} else {
		print(ptr[:])   // want "nil dereference in slice operation"
		*ptr = [3]int{} // want "nil dereference in store"
		print(*ptr)     // want "nil dereference in load"

		if ptr != nil { // want "impossible condition: nil != nil"
			// Dominated by ptr==nil and ptr!=nil,
			// this block is unreachable.
			// We do not report errors within it.
			print(*ptr)
		}
	}

	if i != nil {
		print(i.(interface{ f() }))
	} else {
		print(i.(interface{ f() })) // want "nil dereference in type assertion"
	}
}

func g() error

func f3() error {
	err := g()
	if err != nil {
		return err
	}
	if err != nil && err.Error() == "foo" { // want "impossible condition: nil != nil"
		print(0)
	}
	ch := make(chan int)
	if ch == nil { // want "impossible condition: non-nil == nil"
		print(0)
	}
	if ch != nil { // want "tautological condition: non-nil != nil"
		print(0)
	}
	return nil
}

func h(err error, b bool) {
	if err != nil && b {
		return
	} else if err != nil {
		panic(err)
	}
}

func i(*int) error {
	for {
		if err := g(); err != nil {
			return err
		}
	}
}
