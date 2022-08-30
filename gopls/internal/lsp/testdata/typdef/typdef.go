package typdef

type Struct struct { //@item(Struct, "Struct", "struct{...}", "struct")
	Field string
}

type Int int //@item(Int, "Int", "int", "type")

func _() {
	var (
		value Struct
		point *Struct
	)
	_ = value //@typdef("value", Struct)
	_ = point //@typdef("point", Struct)

	var (
		array   [3]Struct
		slice   []Struct
		ch      chan Struct
		complex [3]chan *[5][]Int
	)
	_ = array   //@typdef("array", Struct)
	_ = slice   //@typdef("slice", Struct)
	_ = ch      //@typdef("ch", Struct)
	_ = complex //@typdef("complex", Int)

	var s struct {
		x struct {
			xx struct {
				field1 []Struct
				field2 []Int
			}
		}
	}
	s.x.xx.field1 //@typdef("field1", Struct)
	s.x.xx.field2 //@typdef("field2", Int)
}

func F1() Int                              { return 0 }
func F2() (Int, float64)                   { return 0, 0 }
func F3() (Struct, int, bool, error)       { return Struct{}, 0, false, nil }
func F4() (**int, Int, bool, *error)       { return nil, Struct{}, false, nil }
func F5() (int, float64, error, Struct)    { return 0, 0, nil, Struct{} }
func F6() (int, float64, ***Struct, error) { return 0, 0, nil, nil }

func _() {
	F1() //@typdef("F1", Int)
	F2() //@typdef("F2", Int)
	F3() //@typdef("F3", Struct)
	F4() //@typdef("F4", Int)
	F5() //@typdef("F5", Struct)
	F6() //@typdef("F6", Struct)

	f := func() Int { return 0 }
	f() //@typdef("f", Int)
}

// https://github.com/golang/go/issues/38589#issuecomment-620350922
func _() {
	type myFunc func(int) Int //@item(myFunc, "myFunc", "func", "type")

	var foo myFunc
	bar := foo() //@typdef("foo", myFunc)
}
