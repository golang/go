package extract

import "strconv"

func _() {
	x0 := append([]int{}, 1) //@suggestedfix("append([]int{}, 1)", "refactor.extract")
	str := "1"
	b, err := strconv.Atoi(str) //@suggestedfix("strconv.Atoi(str)", "refactor.extract")
}
