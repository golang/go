package depBase

var V int = 1

var HasMask []string = []string{"hi"}

type HasProg struct {
	array [1024]*byte
}

type Dep struct {
	X int
}

func F() int {
	return V
}
