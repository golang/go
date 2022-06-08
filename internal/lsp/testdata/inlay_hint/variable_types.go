package inlayHint //@inlayHint("package")

func assignTypes() {
	i, j := 0, len([]string{})-1
	println(i, j)
}

func rangeTypes() {
	for k, v := range []string{} {
		println(k, v)
	}
}

func funcLitType() {
	myFunc := func(a string) string { return "" }
}

func compositeLitType() {
	foo := map[string]interface{}{"": ""}
}
