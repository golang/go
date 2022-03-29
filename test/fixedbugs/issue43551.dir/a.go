package a

type S struct {
	a Key
}

func (s S) A() Key {
	return s.a
}

type Key struct {
	key int64
}
