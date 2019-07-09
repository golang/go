package refs

type i int //@mark(typeInt, "int"),refs("int", typeInt, argInt, returnInt)

func _(_ int) []bool { //@mark(argInt, "int")
	return nil
}

func _(_ string) int { //@mark(returnInt, "int")
	return 0
}
