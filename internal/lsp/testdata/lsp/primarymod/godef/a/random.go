package a

func Random() int { //@Random
	y := 6 + 7
	return y
}

func Random2(y int) int { //@Random2,mark(RandomParamY, "y")
	return y //@godef("y", RandomParamY)
}

type Pos struct {
	x, y int //@mark(PosX, "x"),mark(PosY, "y")
}

func (p *Pos) Sum() int { //@mark(PosSum, "Sum")
	return p.x + p.y //@godef("x", PosX)
}

func _() {
	var p Pos
	_ = p.Sum() //@godef("()", PosSum)
}
