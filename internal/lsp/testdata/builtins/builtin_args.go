package builtins

func _() {
	var (
		aSlice    []int          //@item(builtinSlice, "aSlice", "[]int", "var")
		aMap      map[string]int //@item(builtinMap, "aMap", "map[string]int", "var")
		aString   string         //@item(builtinString, "aString", "string", "var")
		aArray    [0]int         //@item(builtinArray, "aArray", "[0]int", "var")
		aArrayPtr *[0]int        //@item(builtinArrayPtr, "aArrayPtr", "*[0]int", "var")
		aChan     chan int       //@item(builtinChan, "aChan", "chan int", "var")
		aPtr      *int           //@item(builtinPtr, "aPtr", "*int", "var")
		aInt      int            //@item(builtinInt, "aInt", "int", "var")
	)

	type (
		aSliceType []int          //@item(builtinSliceType, "aSliceType", "[]int", "type")
		aChanType  chan int       //@item(builtinChanType, "aChanType", "chan int", "type")
		aMapType   map[string]int //@item(builtinMapType, "aMapType", "map[string]int", "type")
	)

	close() //@rank(")", builtinChan, builtinSlice)

	append() //@rank(")", builtinSlice, builtinChan)

	var _ []byte = append([]byte(nil), ""...) //@rank(") //")

	copy()           //@rank(")", builtinSlice, builtinChan)
	copy(aSlice, aS) //@rank(")", builtinSlice, builtinString)
	copy(aS, aSlice) //@rank(",", builtinSlice, builtinString)

	delete()         //@rank(")", builtinMap, builtinChan)
	delete(aMap, aS) //@rank(")", builtinString, builtinSlice)

	aMapFunc := func() map[int]int { //@item(builtinMapFunc, "aMapFunc", "func() map[int]int", "var")
		return nil
	}
	delete() //@rank(")", builtinMapFunc, builtinSlice)

	len() //@rank(")", builtinSlice, builtinInt),rank(")", builtinMap, builtinInt),rank(")", builtinString, builtinInt),rank(")", builtinArray, builtinInt),rank(")", builtinArrayPtr, builtinPtr),rank(")", builtinChan, builtinInt)

	cap() //@rank(")", builtinSlice, builtinMap),rank(")", builtinArray, builtinString),rank(")", builtinArrayPtr, builtinPtr),rank(")", builtinChan, builtinInt)

	make()              //@rank(")", builtinMapType, int),rank(")", builtinChanType, int),rank(")", builtinSliceType, int),rank(")", builtinMapType, int)
	make(aSliceType, a) //@rank(")", builtinInt, builtinSlice)

	var _ []int = make() //@rank(")", builtinSliceType, builtinMapType)

	type myStruct struct{}  //@item(builtinStructType, "myStruct", "struct{...}", "struct")
	var _ *myStruct = new() //@rank(")", builtinStructType, int)

	for k := range a { //@rank(" {", builtinSlice, builtinInt),rank(" {", builtinString, builtinInt),rank(" {", builtinChan, builtinInt),rank(" {", builtinArray, builtinInt),rank(" {", builtinArrayPtr, builtinInt),rank(" {", builtinMap, builtinInt),
	}

	for k, v := range a { //@rank(" {", builtinSlice, builtinChan)
	}

	<-a //@rank(" //", builtinChan, builtinInt)
}
