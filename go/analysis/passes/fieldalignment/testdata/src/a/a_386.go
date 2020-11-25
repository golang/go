package a

type PointerGood struct {
	P   *int
	buf [1000]uintptr
}

type PointerBad struct { // want "struct with 4004 pointer bytes could be 4"
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

type PointerSortaBad struct { // want "struct with 16 pointer bytes could be 12"
	a struct {
		p *int
		q [2]uintptr
	}
	b struct {
		p *int
		q uintptr
	}
}

type MultiField struct { // want "struct of size 20 could be 12"
	b      bool
	i1, i2 int
	a3     [3]bool
	_      [0]func()
}
