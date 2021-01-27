package a

type T1 struct{ x int }

type T2 struct {
	x int
	y int
}

type T3 struct{ y *T1 }

func BadWrites() {
	// Test struct field writes.
	var s1 T1
	s1.x = 10 // want "unused write to field x"

	// Test array writes.
	var s2 [10]int
	s2[1] = 10 // want "unused write to array index 1:int"

	// Test range variables of struct type.
	s3 := []T1{T1{x: 100}}
	for i, v := range s3 {
		v.x = i // want "unused write to field x"
	}

	// Test the case where a different field is read after the write.
	s4 := []T2{T2{x: 1, y: 2}}
	for i, v := range s4 {
		v.x = i // want "unused write to field x"
		_ = v.y
	}
}

func (t T1) BadValueReceiverWrite(v T2) {
	t.x = 10 // want "unused write to field x"
	v.y = 20 // want "unused write to field y"
}

func GoodWrites(m map[int]int) {
	// A map is copied by reference such that a write will affect the original map.
	m[1] = 10

	// Test struct field writes.
	var s1 T1
	s1.x = 10
	print(s1.x)

	// Test array writes.
	var s2 [10]int
	s2[1] = 10
	// Current the checker doesn't distinguish index 1 and index 2.
	_ = s2[2]

	// Test range variables of struct type.
	s3 := []T1{T1{x: 100}}
	for i, v := range s3 { // v is a copy
		v.x = i
		_ = v.x // still a usage
	}

	// Test an object with multiple fields.
	o := &T2{x: 10, y: 20}
	print(o)

	// Test an object of embedded struct/pointer type.
	t1 := &T1{x: 10}
	t2 := &T3{y: t1}
	print(t2)
}

func (t *T1) GoodPointerReceiverWrite(v *T2) {
	t.x = 10
	v.y = 20
}
