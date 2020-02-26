package multireturn

func f0() {} //@item(multiF0, "f0", "func()", "func")

func f1(int) int { return 0 } //@item(multiF1, "f1", "func(int) int", "func")

func f2(int, int) (int, int) { return 0, 0 } //@item(multiF2, "f2", "func(int, int) (int, int)", "func")

func f2Str(string, string) (string, string) { return "", "" } //@item(multiF2Str, "f2Str", "func(string, string) (string, string)", "func")

func f3(int, int, int) (int, int, int) { return 0, 0, 0 } //@item(multiF3, "f3", "func(int, int, int) (int, int, int)", "func")

func _() {
	_ := f //@rank(" //", multiF1, multiF2)

	_, _ := f //@rank(" //", multiF2, multiF0),rank(" //", multiF1, multiF0)

	_, _ := _, f //@rank(" //", multiF1, multiF2),rank(" //", multiF1, multiF0)

	_, _ := f, abc //@rank(", abc", multiF1, multiF2)

	f1()     //@rank(")", multiF1, multiF0)
	f1(f)    //@rank(")", multiF1, multiF2)
	f2(f)    //@rank(")", multiF2, multiF3),rank(")", multiF1, multiF3)
	f2(1, f) //@rank(")", multiF1, multiF2),rank(")", multiF1, multiF0)
	f2Str()  //@rank(")", multiF2Str, multiF2)

	var i int
	i, _ := f //@rank(" //", multiF2, multiF2Str)

	var s string
	_, s := f //@rank(" //", multiF2Str, multiF2)

	banana, s = f //@rank(" //", multiF2, multiF3)

	var variadic func(int, ...int)
	variadic() //@rank(")", multiF1, multiF0),rank(")", multiF2, multiF0),rank(")", multiF3, multiF0)
}
