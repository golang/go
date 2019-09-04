package snippets

func _() {
	[]int{}        //@item(litIntSlice, "[]int{}", "", "var")
	make([]int, 0) //@item(makeIntSlice, "make([]int, 0)", "", "func")

	var slice []int
	slice = i //@snippet(" //", litIntSlice, "[]int{$0\\}", "[]int{$0\\}")
	slice = m //@snippet(" //", makeIntSlice, "make([]int, ${1:})", "make([]int, ${1:0})")
}

func _() {
	type namedInt []int

	namedInt{}        //@item(litNamedSlice, "namedInt{}", "", "var")
	make(namedInt, 0) //@item(makeNamedSlice, "make(namedInt, 0)", "", "func")

	var namedSlice namedInt
	namedSlice = n //@snippet(" //", litNamedSlice, "namedInt{$0\\}", "namedInt{$0\\}")
	namedSlice = m //@snippet(" //", makeNamedSlice, "make(namedInt, ${1:})", "make(namedInt, ${1:0})")
}

func _() {
	make(chan int) //@item(makeChan, "make(chan int)", "", "func")

	var ch chan int
	ch = m //@snippet(" //", makeChan, "make(chan int)", "make(chan int)")
}

func _() {
	map[string]struct{}{}     //@item(litMap, "map[string]struct{}{}", "", "var")
	make(map[string]struct{}) //@item(makeMap, "make(map[string]struct{})", "", "func")

	var m map[string]struct{}
	m = m //@snippet(" //", litMap, "map[string]struct{\\}{$0\\}", "map[string]struct{\\}{$0\\}")
	m = m //@snippet(" //", makeMap, "make(map[string]struct{\\})", "make(map[string]struct{\\})")

	struct{}{} //@item(litEmptyStruct, "struct{}{}", "", "var")

	m["hi"] = s //@snippet(" //", litEmptyStruct, "struct{\\}{\\}", "struct{\\}{\\}")
}

func _() {
	type myStruct struct{ i int }

	myStruct{}  //@item(litStruct, "myStruct{}", "", "var")
	&myStruct{} //@item(litStructPtr, "&myStruct{}", "", "var")

	var ms myStruct
	ms = m //@snippet(" //", litStruct, "myStruct{$0\\}", "myStruct{$0\\}")

	var msPtr *myStruct
	msPtr = m //@snippet(" //", litStructPtr, "&myStruct{$0\\}", "&myStruct{$0\\}")

	msPtr = &m //@snippet(" //", litStruct, "myStruct{$0\\}", "myStruct{$0\\}")
}

type myImpl struct{}

func (myImpl) foo() {}

func (*myImpl) bar() {}

func _() {
	type myIntf interface {
		foo()
	}

	myImpl{} //@item(litImpl, "myImpl{}", "", "var")

	var mi myIntf
	mi = m //@snippet(" //", litImpl, "myImpl{\\}", "myImpl{\\}")

	// only satisfied by pointer to myImpl
	type myPtrIntf interface {
		bar()
	}

	&myImpl{} //@item(litImplPtr, "&myImpl{}", "", "var")

	var mpi myPtrIntf
	mpi = m //@snippet(" //", litImplPtr, "&myImpl{\\}", "&myImpl{\\}")
}

func _() {
	var s struct{ i []int } //@item(litSliceField, "i", "[]int", "field")
	var foo []int
	// no literal completions after selector
	foo = s.i //@complete(" //", litSliceField)
}

func _() {
	type myStruct struct{ i int } //@item(litStructType, "myStruct", "struct{...}", "struct")

	foo := func(s string, args ...myStruct) {}
	// Don't give literal slice candidate for variadic arg.
	foo("", myStruct) //@complete(")", litStructType)
}
