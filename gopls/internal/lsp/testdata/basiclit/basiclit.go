package basiclit

func _() {
	var a int // something for lexical completions

	_ = "hello." //@complete(".")

	_ = 1 //@complete(" //")

	_ = 1. //@complete(".")

	_ = 'a' //@complete("' ")
}
