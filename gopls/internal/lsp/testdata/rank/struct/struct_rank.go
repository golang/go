package struct_rank

type foo struct {
	c int //@item(c_rank, "c", "int", "field")
	b int //@item(b_rank, "b", "int", "field")
	a int //@item(a_rank, "a", "int", "field")
}

func f() {
	foo := foo{} //@rank("}", c_rank, b_rank, a_rank)
}
