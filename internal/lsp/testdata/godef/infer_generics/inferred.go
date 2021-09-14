package inferred

func app[S interface{ ~[]E }, E any](s S, e E) S {
	return append(s, e)
}

func _() {
	_ = app[[]int]             //@mark(constrInfer, "app"),hoverdef("app", constrInfer)
	_ = app[[]int, int]        //@mark(instance, "app"),hoverdef("app", instance)
	_ = app[[]int]([]int{}, 0) //@mark(partialInfer, "app"),hoverdef("app", partialInfer)
	_ = app([]int{}, 0)        //@mark(argInfer, "app"),hoverdef("app", argInfer)
}
