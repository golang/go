package statements

func _() {
	type mySlice []int

	var (
		abc    []int   //@item(stmtABC, "abc", "[]int", "var")
		abcdef mySlice //@item(stmtABCDEF, "abcdef", "mySlice", "var")
	)

	/* abcdef = append(abcdef, ) */ //@item(stmtABCDEFAssignAppend, "abcdef = append(abcdef, )", "", "func")

	// don't offer "abc = append(abc, )" because "abc" isn't necessarily
	// better than "abcdef".
	abc //@complete(" //", stmtABC, stmtABCDEF)

	abcdef //@complete(" //", stmtABCDEF, stmtABCDEFAssignAppend)

	/* append(abc, ) */ //@item(stmtABCAppend, "append(abc, )", "", "func")

	abc = app //@snippet(" //", stmtABCAppend, "append(abc, ${1:})", "append(abc, ${1:})")
}

func _() {
	var s struct{ xyz []int }

	/* xyz = append(s.xyz, ) */ //@item(stmtXYZAppend, "xyz = append(s.xyz, )", "", "func")

	s.x //@snippet(" //", stmtXYZAppend, "xyz = append(s.xyz, ${1:})", "xyz = append(s.xyz, ${1:})")

	/* s.xyz = append(s.xyz, ) */ //@item(stmtDeepXYZAppend, "s.xyz = append(s.xyz, )", "", "func")

	sx //@snippet(" //", stmtDeepXYZAppend, "s.xyz = append(s.xyz, ${1:})", "s.xyz = append(s.xyz, ${1:})")
}

func _() {
	var foo [][]int

	/* append(foo[0], ) */ //@item(stmtFooAppend, "append(foo[0], )", "", "func")

	foo[0] = app //@complete(" //"),snippet(" //", stmtFooAppend, "append(foo[0], ${1:})", "append(foo[0], ${1:})")
}
