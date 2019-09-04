package refs

type i int //@mark(typeInt, "int"),refs("int", typeInt, argInt, returnInt)

func _(_ int) []bool { //@mark(argInt, "int")
	return nil
}

func _(_ string) int { //@mark(returnInt, "int")
	return 0
}

var q string //@mark(declQ, "q"),refs("q", declQ, assignQ, bobQ)

func _() {
	q = "hello" //@mark(assignQ, "q")
	bob := func(_ string) {}
	bob(q) //@mark(bobQ, "q")
}
