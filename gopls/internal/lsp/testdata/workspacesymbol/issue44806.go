package main

type T struct{}

// We should accept all valid receiver syntax when scanning symbols.
func (*(T)) m1() {}
func (*T) m2()   {}
func (T) m3()    {}
func ((T)) m4()    {}
func ((*T)) m5()   {}
