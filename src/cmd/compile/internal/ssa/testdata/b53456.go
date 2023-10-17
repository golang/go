package main

type T struct {
	m map[int]int
}

func main() {
	t := T{
		m: make(map[int]int),
	}
	t.Inc(5)
	t.Inc(7)
}

func (s *T) Inc(key int) {
	v := s.m[key] // break, line 16
	v++
	s.m[key] = v // also here
}
