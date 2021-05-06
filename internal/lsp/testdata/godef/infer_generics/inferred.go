package inferred

func app[S interface{ ~[]E }, E any](s S, e E) S {
	return append(s, e)
}

func _() {
	_ = app[[]int]             //@mark(constrInfer, "app"),hover("app", constrInfer)
	_ = app[[]int, int]        //@mark(instance, "app"),hover("app", instance)
	_ = app[[]int]([]int{}, 0) //@mark(partialInfer, "app"),hover("app", partialInfer)
	_ = app([]int{}, 0)        //@mark(argInfer, "app"),hover("app", argInfer)
}
