package extract

func _() {
	var _ = 1 + 2 //@suggestedfix("1", "refactor.extract")
	var _ = 3 + 4 //@suggestedfix("3 + 4", "refactor.extract")
}
