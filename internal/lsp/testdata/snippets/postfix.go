package snippets

// These tests check that postfix completions do and do not show up in
// certain cases. Tests for the postfix completion contents are under
// regtest.

func _() {
	/* append! */ //@item(postfixAppend, "append!", "append and re-assign slice", "snippet")
	var foo []int
	foo.append //@rank(" //", postfixAppend)

	[]int{}.append //@complete(" //")

	[]int{}.last //@complete(" //")

	/* copy! */ //@item(postfixCopy, "copy!", "duplicate slice", "snippet")

	foo.copy //@rank(" //", postfixCopy)

	var s struct{ i []int }
	s.i.copy //@rank(" //", postfixCopy)

	var _ []int = s.i.copy //@complete(" //")

	var blah func() []int
	blah().append //@complete(" //")
}

func _() {
	/* append! */ //@item(postfixAppend, "append!", "append and re-assign slice", "snippet")
	/* last! */ //@item(postfixLast, "last!", "s[len(s)-1]", "snippet")
	/* print! */ //@item(postfixPrint, "print!", "print to stdout", "snippet")
	/* range! */ //@item(postfixRange, "range!", "range over slice", "snippet")
	/* reverse! */ //@item(postfixReverse, "reverse!", "reverse slice", "snippet")
	/* sort! */ //@item(postfixSort, "sort!", "sort.Slice()", "snippet")
	/* var! */ //@item(postfixVar, "var!", "assign to variable", "snippet")

	var foo []int
	foo. //@complete(" //", postfixAppend, postfixCopy, postfixLast, postfixPrint, postfixRange, postfixReverse, postfixSort, postfixVar)

		foo = nil
}
