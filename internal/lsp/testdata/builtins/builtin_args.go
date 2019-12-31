package builtins

func _() {
	var (
		slice_    []int          //@item(builtinSlice, "slice_", "[]int", "var")
		map_      map[string]int //@item(builtinMap, "map_", "map[string]int", "var")
		string_   string         //@item(builtinString, "string_", "string", "var")
		array_    [0]int         //@item(builtinArray, "array_", "[0]int", "var")
		arrayPtr_ *[0]int        //@item(builtinArrayPtr, "arrayPtr_", "*[0]int", "var")
		chan_     chan int       //@item(builtinChan, "chan_", "chan int", "var")
		ptr_      *int           //@item(builtinPtr, "ptr_", "*int", "var")
		int_      int            //@item(builtinInt, "int_", "int", "var")
	)

	close() //@rank(")", builtinChan, builtinSlice)

	append() //@rank(")", builtinSlice, builtinChan)

	copy()          //@rank(")", builtinSlice, builtinChan)
	copy(slice_, s) //@rank(")", builtinSlice, builtinString)
	copy(s, slice_) //@rank(",", builtinSlice, builtinString)

	delete()        //@rank(")", builtinMap, builtinChan)
	delete(map_, s) //@rank(")", builtinString, builtinSlice)

	len() //@rank(")", builtinSlice, builtinInt),rank(")", builtinMap, builtinInt),rank(")", builtinString, builtinInt),rank(")", builtinArray, builtinInt),rank(")", builtinArrayPtr, builtinPtr),rank(")", builtinChan, builtinInt)

	cap() //@rank(")", builtinSlice, builtinMap),rank(")", builtinArray, builtinString),rank(")", builtinArrayPtr, builtinPtr),rank(")", builtinChan, builtinInt)

	make() //@rank(")", builtinMap, builtinInt),rank(")", builtinChan, builtinInt),rank(")", builtinSlice, builtinInt)

	var _ []int = make() //@rank(")", builtinSlice, builtinMap)

	type myStruct struct{}  //@item(builtinStructType, "myStruct", "struct{...}", "struct")
	new()                   //@rank(")", builtinStructType, builtinInt)
	var _ *myStruct = new() //@rank(")", builtinStructType, int)
}
