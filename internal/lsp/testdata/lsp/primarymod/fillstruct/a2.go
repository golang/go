package fillstruct

type typedStruct struct {
	m  map[string]int
	s  []int
	c  chan int
	c1 <-chan int
	a  [2]string
}

var _ = typedStruct{} //@suggestedfix("}", "refactor.rewrite")

type funStruct struct {
	fn func(i int) int
}

var _ = funStruct{} //@suggestedfix("}", "refactor.rewrite")

type funStructCompex struct {
	fn func(i int, s string) (string, int)
}

var _ = funStructCompex{} //@suggestedfix("}", "refactor.rewrite")

type funStructEmpty struct {
	fn func()
}

var _ = funStructEmpty{} //@suggestedfix("}", "refactor.rewrite")
