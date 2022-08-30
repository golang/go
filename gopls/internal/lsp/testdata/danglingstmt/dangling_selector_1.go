package danglingstmt

func _() {
	x. //@rank(" //", danglingI)
}

var x struct { i int } //@item(danglingI, "i", "int", "field")
