package a

type Package struct {
	name string
}

type Future struct {
	result chan struct {
		*Package
	}
}

func (t *Future) Result() *Package {
	result := <-t.result
	t.result <- result
	return result.Package
}
