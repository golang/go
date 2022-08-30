package noresult

func hello[T any]() {
	var z T
	return z // want `no result values expected|too many return values`
}
