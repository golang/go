//go:build go1.18
// +build go1.18

package typeparams

func one[a int | string]()            {}
func two[a int | string, b float64 | int]() {}

func _() {
	one[]() //@rank("]", string, float64)
	two[]() //@rank("]", int, float64)
	two[int, f]() //@rank("]", float64, float32)
}

func slices[a []int | []float64]() {} //@item(tpInts, "[]int", "[]int", "type"),item(tpFloats, "[]float64", "[]float64", "type")

func _() {
	slices[]() //@rank("]", tpInts),rank("]", tpFloats)
}

type s[a int | string] struct{}

func _() {
	s[]{} //@rank("]", int, float64)
}

func takesGeneric[a int | string](s[a]) {
	"s[a]{}" //@item(tpInScopeLit, "s[a]{}", "", "var")
	takesGeneric() //@rank(")", tpInScopeLit),snippet(")", tpInScopeLit, "s[a]{\\}", "s[a]{\\}")
}

func _() {
	s[int]{} //@item(tpInstLit, "s[int]{}", "", "var")
	takesGeneric[int]() //@rank(")", tpInstLit),snippet(")", tpInstLit, "s[int]{\\}", "s[int]{\\}")

	"s[...]{}" //@item(tpUninstLit, "s[...]{}", "", "var")
	takesGeneric() //@rank(")", tpUninstLit),snippet(")", tpUninstLit, "s[${1:}]{\\}", "s[${1:a}]{\\}")
}

func returnTP[A int | float64](a A) A { //@item(returnTP, "returnTP", "something", "func")
	return a
}

func _() {
	var _ int = returnTP //@snippet(" //", returnTP, "returnTP[${1:}](${2:})", "returnTP[${1:A int|float64}](${2:a A})")

	var aa int //@item(tpInt, "aa", "int", "var")
	var ab float64 //@item(tpFloat, "ab", "float64", "var")
	returnTP[int](a) //@rank(")", tpInt, tpFloat)
}

func takesFunc[T any](func(T) T) {
	var _ func(t T) T = f //@snippet(" //", tpLitFunc, "func(t T) T {$0\\}", "func(t T) T {$0\\}")
}

func _() {
	_ = "func(...) {}" //@item(tpLitFunc, "func(...) {}", "", "var")
	takesFunc() //@snippet(")", tpLitFunc, "func(${1:}) ${2:} {$0\\}", "func(${1:t} ${2:T}) ${3:T} {$0\\}")
	takesFunc[int]() //@snippet(")", tpLitFunc, "func(i int) int {$0\\}", "func(${1:i} int) int {$0\\}")
}
