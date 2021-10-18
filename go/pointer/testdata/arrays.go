//go:build ignore
// +build ignore

package main

var unknown bool // defeat dead-code elimination

var a, b int

func array1() {
	sliceA := make([]*int, 10) // @line a1make
	sliceA[0] = &a

	var sliceB []*int
	sliceB = append(sliceB, &b) // @line a1append

	print(sliceA)    // @pointsto makeslice@a1make:16
	print(sliceA[0]) // @pointsto command-line-arguments.a

	print(sliceB)      // @pointsto append@a1append:17
	print(sliceB[100]) // @pointsto command-line-arguments.b
}

func array2() {
	sliceA := make([]*int, 10) // @line a2make
	sliceA[0] = &a

	sliceB := sliceA[:]

	print(sliceA)    // @pointsto makeslice@a2make:16
	print(sliceA[0]) // @pointsto command-line-arguments.a

	print(sliceB)    // @pointsto makeslice@a2make:16
	print(sliceB[0]) // @pointsto command-line-arguments.a
}

func array3() {
	a := []interface{}{"", 1}
	b := []interface{}{true, func() {}}
	print(a[0]) // @types string | int
	print(b[0]) // @types bool | func()
}

// Test of append, copy, slice.
func array4() {
	var s2 struct { // @line a4L0
		a [3]int
		b struct{ c, d int }
	}
	var sl1 = make([]*int, 10) // @line a4make
	var someint int            // @line a4L1
	sl1[1] = &someint
	sl2 := append(sl1, &s2.a[1]) // @line a4append1
	print(sl1)                   // @pointsto makeslice@a4make:16
	print(sl2)                   // @pointsto append@a4append1:15 | makeslice@a4make:16
	print(sl1[0])                // @pointsto someint@a4L1:6 | s2.a[*]@a4L0:6
	print(sl2[0])                // @pointsto someint@a4L1:6 | s2.a[*]@a4L0:6

	// In z=append(x,y) we should observe flow from y[*] to x[*].
	var sl3 = make([]*int, 10) // @line a4L2
	_ = append(sl3, &s2.a[1])
	print(sl3)    // @pointsto makeslice@a4L2:16
	print(sl3[0]) // @pointsto s2.a[*]@a4L0:6

	var sl4 = []*int{&a} // @line a4L3
	sl4a := append(sl4)  // @line a4L4
	print(sl4a)          // @pointsto slicelit@a4L3:18 | append@a4L4:16
	print(&sl4a[0])      // @pointsto slicelit[*]@a4L3:18 | append[*]@a4L4:16
	print(sl4a[0])       // @pointsto command-line-arguments.a

	var sl5 = []*int{&b} // @line a4L5
	copy(sl5, sl4)
	print(sl5)     // @pointsto slicelit@a4L5:18
	print(&sl5[0]) // @pointsto slicelit[*]@a4L5:18
	print(sl5[0])  // @pointsto command-line-arguments.b | command-line-arguments.a

	var sl6 = sl5[:0]
	print(sl6)     // @pointsto slicelit@a4L5:18
	print(&sl6[0]) // @pointsto slicelit[*]@a4L5:18
	print(sl6[0])  // @pointsto command-line-arguments.b | command-line-arguments.a
}

func array5() {
	var arr [2]*int
	arr[0] = &a
	arr[1] = &b

	var n int
	print(arr[n]) // @pointsto command-line-arguments.a | command-line-arguments.b
}

func main() {
	array1()
	array2()
	array3()
	array4()
	array5()
}
