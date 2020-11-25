package a

type PointerGood struct {
	P   *int
	buf [1000]uintptr
}

type PointerBad struct { // want "struct with 8008 pointer bytes could be 8"
	buf [1000]uintptr
	P   *int
}

type PointerSorta struct {
	a struct {
		p *int
		q uintptr
	}
	b struct {
		p *int
		q [2]uintptr
	}
}

type PointerSortaBad struct { // want "struct with 32 pointer bytes could be 24"
	a struct {
		p *int
		q [2]uintptr
	}
	b struct {
		p *int
		q uintptr
	}
}

type MultiField struct { // want "struct of size 40 could be 24"
	b      bool
	i1, i2 int
	a3     [3]bool
	_      [0]func()
}
