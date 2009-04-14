// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package reflect

import (
	"reflect";
	"testing"
)

var doprint bool = false

func is_digit(c uint8) bool {
	return '0' <= c && c <= '9'
}

// streq, but '@' in t matches a string of digits
func match(s, t string) bool {
	for i, j := 0, 0; i < len(s) && j < len(t); i, j = i+1, j+1 {
		if s[i] == t[j] {
			continue
		}
		if is_digit(s[i]) && t[j] == '@' {
			for is_digit(s[i+1]) {
				i++
			}
		} else {
			return false
		}
	}
	return true;
}

func assert(s, t string) {
	if doprint {
		println(t)
	}
	if !match(s, t) {
		panicln(s, t)
	}
}

func typedump(s, t string) {
	typ := reflect.ParseTypeString("", s);
	assert(reflect.typeToString(typ, true), t);
}

func valuedump(s, t string) {
	typ := reflect.ParseTypeString("", s);
	v := reflect.NewInitValue(typ);
	if v == nil {
		panicln("valuedump", s);
	}
	switch v.Kind() {
	case reflect.IntKind:
		v.(reflect.IntValue).Set(132);
	case reflect.Int8Kind:
		v.(reflect.Int8Value).Set(8);
	case reflect.Int16Kind:
		v.(reflect.Int16Value).Set(16);
	case reflect.Int32Kind:
		v.(reflect.Int32Value).Set(32);
	case reflect.Int64Kind:
		v.(reflect.Int64Value).Set(64);
	case reflect.UintKind:
		v.(reflect.UintValue).Set(132);
	case reflect.Uint8Kind:
		v.(reflect.Uint8Value).Set(8);
	case reflect.Uint16Kind:
		v.(reflect.Uint16Value).Set(16);
	case reflect.Uint32Kind:
		v.(reflect.Uint32Value).Set(32);
	case reflect.Uint64Kind:
		v.(reflect.Uint64Value).Set(64);
	case reflect.FloatKind:
		v.(reflect.FloatValue).Set(3200.0);
	case reflect.Float32Kind:
		v.(reflect.Float32Value).Set(32.1);
	case reflect.Float64Kind:
		v.(reflect.Float64Value).Set(64.2);
	case reflect.StringKind:
		v.(reflect.StringValue).Set("stringy cheese");
	case reflect.BoolKind:
		v.(reflect.BoolValue).Set(true);
	}
	assert(reflect.valueToString(v), t);
}

type T struct { a int; b float64; c string; d *int }

func TestAll(tt *testing.T) {	// TODO(r): wrap up better
	var s string;
	var t reflect.Type;

	// Types
	typedump("missing", "$missing$");
	typedump("int", "int");
	typedump("int8", "int8");
	typedump("int16", "int16");
	typedump("int32", "int32");
	typedump("int64", "int64");
	typedump("uint", "uint");
	typedump("uint8", "uint8");
	typedump("uint16", "uint16");
	typedump("uint32", "uint32");
	typedump("uint64", "uint64");
	typedump("float", "float");
	typedump("float32", "float32");
	typedump("float64", "float64");
	typedump("int8", "int8");
	typedump("whoknows.whatsthis", "$missing$");
	typedump("**int8", "**int8");
	typedump("**P.integer", "**P.integer");
	typedump("[32]int32", "[32]int32");
	typedump("[]int8", "[]int8");
	typedump("map[string]int32", "map[string]int32");
	typedump("chan<-string", "chan<-string");
	typedump("struct {c chan *int32; d float32}", "struct{c chan*int32; d float32}");
	typedump("func(a int8, b int32)", "func(a int8, b int32)");
	typedump("struct {c func(? chan *P.integer, ? *int8)}", "struct{c func(chan*P.integer, *int8)}");
	typedump("struct {a int8; b int32}", "struct{a int8; b int32}");
	typedump("struct {a int8; b int8; b int32}", "struct{a int8; b int8; b int32}");
	typedump("struct {a int8; b int8; c int8; b int32}", "struct{a int8; b int8; c int8; b int32}");
	typedump("struct {a int8; b int8; c int8; d int8; b int32}", "struct{a int8; b int8; c int8; d int8; b int32}");
	typedump("struct {a int8; b int8; c int8; d int8; e int8; b int32}", "struct{a int8; b int8; c int8; d int8; e int8; b int32}");
	typedump("struct {a int8 \"hi there\"; }", "struct{a int8 \"hi there\"}");
	typedump("struct {a int8 \"hi \\x00there\\t\\n\\\"\\\\\"; }", "struct{a int8 \"hi \\x00there\\t\\n\\\"\\\\\"}");
	typedump("struct {f func(args ...)}", "struct{f func(args ...)}");
	typedump("interface { a(? func(? func(? int) int) func(? func(? int)) int); b() }", "interface{a (func(func(int)(int))(func(func(int))(int))); b ()}");

	// Values
	valuedump("int8", "8");
	valuedump("int16", "16");
	valuedump("int32", "32");
	valuedump("int64", "64");
	valuedump("uint8", "8");
	valuedump("uint16", "16");
	valuedump("uint32", "32");
	valuedump("uint64", "64");
	valuedump("float32", "32.1");
	valuedump("float64", "64.2");
	valuedump("string", "stringy cheese");
	valuedump("bool", "true");
	valuedump("*int8", "*int8(0)");
	valuedump("**int8", "**int8(0)");
	valuedump("[5]int32", "[5]int32{0, 0, 0, 0, 0}");
	valuedump("**P.integer", "**P.integer(0)");
	valuedump("map[string]int32", "map[string]int32{<can't iterate on maps>}");
	valuedump("chan<-string", "chan<-string");
	valuedump("struct {c chan *int32; d float32}", "struct{c chan*int32; d float32}{chan*int32, 0}");
	valuedump("func(a int8, b int32)", "func(a int8, b int32)(0)");
	valuedump("struct {c func(? chan *P.integer, ? *int8)}", "struct{c func(chan*P.integer, *int8)}{func(chan*P.integer, *int8)(0)}");
	valuedump("struct {a int8; b int32}", "struct{a int8; b int32}{0, 0}");
	valuedump("struct {a int8; b int8; b int32}", "struct{a int8; b int8; b int32}{0, 0, 0}");

	{	var tmp = 123;
		value := reflect.NewValue(tmp);
		assert(reflect.valueToString(value), "123");
	}
	{	var tmp = 123.4;
		value := reflect.NewValue(tmp);
		assert(reflect.valueToString(value), "123.4");
	}
	{
		var tmp = byte(123);
		value := reflect.NewValue(tmp);
		assert(reflect.valueToString(value), "123");
		assert(reflect.typeToString(value.Type(), false), "uint8");
	}
	{	var tmp = "abc";
		value := reflect.NewValue(tmp);
		assert(reflect.valueToString(value), "abc");
	}
	{
		var i int = 7;
		var tmp = &T{123, 456.75, "hello", &i};
		value := reflect.NewValue(tmp);
		assert(reflect.valueToString(value.(reflect.PtrValue).Sub()), "reflect.T{123, 456.75, hello, *int(@)}");
	}
	{
		type C chan *T;	// TODO: should not be necessary
		var tmp = new(C);
		value := reflect.NewValue(tmp);
		assert(reflect.valueToString(value), "*reflect.C·all_test(@)");
	}
//	{
//		type A [10]int;
//		var tmp A = A{1,2,3,4,5,6,7,8,9,10};
//		value := reflect.NewValue(&tmp);
//		assert(reflect.valueToString(value.(reflect.PtrValue).Sub()), "reflect.A·all_test{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}");
//		value.(reflect.PtrValue).Sub().(reflect.ArrayValue).Elem(4).(reflect.IntValue).Set(123);
//		assert(reflect.valueToString(value.(reflect.PtrValue).Sub()), "reflect.A·all_test{1, 2, 3, 4, 123, 6, 7, 8, 9, 10}");
//	}
	{
		type AA []int;
		var tmp = AA{1,2,3,4,5,6,7,8,9,10};
		value := reflect.NewValue(&tmp);	// TODO: NewValue(tmp) too
		assert(reflect.valueToString(value.(reflect.PtrValue).Sub()), "reflect.AA·all_test{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}");
		value.(reflect.PtrValue).Sub().(reflect.ArrayValue).Elem(4).(reflect.IntValue).Set(123);
		assert(reflect.valueToString(value.(reflect.PtrValue).Sub()), "reflect.AA·all_test{1, 2, 3, 4, 123, 6, 7, 8, 9, 10}");
	}

	{
		var ip *int32;
		var i int32 = 1234;
		vip := reflect.NewValue(&ip);
		vi := reflect.NewValue(i);
		vip.(reflect.PtrValue).Sub().(reflect.PtrValue).SetSub(vi);
		if *ip != 1234 {
			panicln("SetSub failure", *ip);
		}
	}

	var pt reflect.PtrType;
	var st reflect.StructType;
	var mt reflect.MapType;
	var at reflect.ArrayType;
	var ct reflect.ChanType;
	var name string;
	var typ reflect.Type;
	var tag string;
	var offset int;

	// Type strings
	t = reflect.ParseTypeString("", "int8");
	assert(t.String(), "int8");

	t = reflect.ParseTypeString("", "*int8");
	assert(t.String(), "*int8");
	pt = t.(reflect.PtrType);
	assert(pt.Sub().String(), "int8");

	t = reflect.ParseTypeString("", "*struct {c chan *int32; d float32}");
	assert(t.String(), "*struct {c chan *int32; d float32}");
	pt = t.(reflect.PtrType);
	assert(pt.Sub().String(), "struct {c chan *int32; d float32}");
	st = pt.Sub().(reflect.StructType);
	name, typ, tag, offset = st.Field(0);
	assert(typ.String(), "chan *int32");
	name, typ, tag, offset = st.Field(1);
	assert(typ.String(), "float32");

	t = reflect.ParseTypeString("", "interface {a() *int}");
	assert(t.String(), "interface {a() *int}");

	t = reflect.ParseTypeString("", "func(a int8, b int32)");
	assert(t.String(), "func(a int8, b int32)");

	t = reflect.ParseTypeString("", "func(a int8, b int32) float");
	assert(t.String(), "func(a int8, b int32) float");

	t = reflect.ParseTypeString("", "func(a int8, b int32) (a float, b float)");
	assert(t.String(), "func(a int8, b int32) (a float, b float)");

	t = reflect.ParseTypeString("", "[32]int32");
	assert(t.String(), "[32]int32");
	at = t.(reflect.ArrayType);
	assert(at.Elem().String(), "int32");

	t = reflect.ParseTypeString("", "map[string]*int32");
	assert(t.String(), "map[string]*int32");
	mt = t.(reflect.MapType);
	assert(mt.Key().String(), "string");
	assert(mt.Elem().String(), "*int32");

	t = reflect.ParseTypeString("", "chan<-string");
	assert(t.String(), "chan<-string");
	ct = t.(reflect.ChanType);
	assert(ct.Elem().String(), "string");

	// make sure tag strings are not part of element type
	t = reflect.ParseTypeString("", "struct{d []uint32 \"TAG\"}");
	st = t.(reflect.StructType);
	name, typ, tag, offset = st.Field(0);
	assert(typ.String(), "[]uint32");

	t = reflect.ParseTypeString("", "[]int32");
	v := reflect.NewSliceValue(t.(ArrayType), 5, 10);
	t1 := reflect.ParseTypeString("", "*[]int32");
	v1 := reflect.NewInitValue(t1);
	if v1 == nil { panic("V1 is nil"); }
	v1.(reflect.PtrValue).SetSub(v);
	a := v1.Interface().(*[]int32);
	println(&a, len(a), cap(a));
	for i := 0; i < len(a); i++ {
		v.Elem(i).(reflect.Int32Value).Set(int32(i));
	}
	for i := 0; i < len(a); i++ {
		println(a[i]);
	}
}

func TestInterfaceGet(t *testing.T) {
	var inter struct { e interface{ } };
	inter.e = 123.456;
	v1 := reflect.NewValue(&inter);
	v2 := v1.(reflect.PtrValue).Sub().(reflect.StructValue).Field(0);
	assert(v2.Type().String(), "interface { }");
	i2 := v2.(reflect.InterfaceValue).Get();
	v3 := reflect.NewValue(i2);
	assert(v3.Type().String(), "float");
}

func TestInterfaceValue(t *testing.T) {
	var inter struct { e interface{ } };
	inter.e = 123.456;
	v1 := reflect.NewValue(&inter);
	v2 := v1.(reflect.PtrValue).Sub().(reflect.StructValue).Field(0);
	assert(v2.Type().String(), "interface { }");
	v3 := v2.(reflect.InterfaceValue).Value();
	assert(v3.Type().String(), "float");
}

func TestFunctionValue(t *testing.T) {
	v := reflect.NewValue(func() {});
	if v.Interface() != v.Interface() {
		t.Fatalf("TestFunction != itself");
	}
	assert(v.Type().String(), "func()");
}

func TestCopyArray(t *testing.T) {
	a := []int{ 1, 2, 3, 4, 10, 9, 8, 7 };
	b := []int{ 11, 22, 33, 44, 1010, 99, 88, 77, 66, 55, 44 };
	c := []int{ 11, 22, 33, 44, 1010, 99, 88, 77, 66, 55, 44 };
	va := NewValue(&a);
	vb := NewValue(&b);
	for i := 0; i < len(b); i++ {
		if b[i] != c[i] {
			t.Fatalf("b != c before test");
		}
	}
	for tocopy := 1; tocopy <= 7; tocopy++ {
		vb.(PtrValue).Sub().(ArrayValue).CopyFrom(va.(PtrValue).Sub().(ArrayValue), tocopy);
		for i := 0; i < tocopy; i++ {
			if a[i] != b[i] {
				t.Errorf("1 tocopy=%d a[%d]=%d, b[%d]=%d",
					tocopy, i, a[i], i, b[i]);
			}
		}
		for i := tocopy; i < len(b); i++ {
			if b[i] != c[i] {
				if i < len(a) {
					t.Errorf("2 tocopy=%d a[%d]=%d, b[%d]=%d, c[%d]=%d",
						tocopy, i, a[i], i, b[i], i, c[i]);
				} else {
					t.Errorf("3 tocopy=%d b[%d]=%d, c[%d]=%d",
						tocopy, i, b[i], i, c[i]);
				}
			} else {
				t.Logf("tocopy=%d elem %d is okay\n", tocopy, i);
			}
		}
	}
}

func TestBigUnnamedStruct(t *testing.T) {
	b := struct{a,b,c,d int64}{1, 2, 3, 4};
	v := NewValue(b);
	b1 := v.Interface().(struct{a,b,c,d int64});
	if b1.a != b.a || b1.b != b.b || b1.c != b.c || b1.d != b.d {
		t.Errorf("NewValue(%v).Interface().(Big) = %v", b, b1);
	}
}

type big struct {
	a, b, c, d, e int64
}
func TestBigStruct(t *testing.T) {
	b := big{1, 2, 3, 4, 5};
	v := NewValue(b);
	b1 := v.Interface().(big);
	if b1.a != b.a || b1.b != b.b || b1.c != b.c || b1.d != b.d || b1.e != b.e {
		t.Errorf("NewValue(%v).Interface().(big) = %v", b, b1);
	}
}

type Basic struct {
	x int;
	y float32
}

type Recursive struct {
	x int;
	r *Recursive
}

type Complex struct {
	a int;
	b [3]*Complex;
	c *string;
	d map[float]float
}

type DeepEqualTest struct {
	a, b interface{};
	eq bool;
}

var deepEqualTests = []DeepEqualTest {
	// Equalities
	DeepEqualTest{ 1, 1, true },
	DeepEqualTest{ int32(1), int32(1), true },
	DeepEqualTest{ 0.5, 0.5, true },
	DeepEqualTest{ float32(0.5), float32(0.5), true },
	DeepEqualTest{ "hello", "hello", true },
	DeepEqualTest{ make([]int, 10), make([]int, 10), true },
	DeepEqualTest{ &[3]int{ 1, 2, 3 }, &[3]int{ 1, 2, 3 }, true },
	DeepEqualTest{ Basic{ 1, 0.5 }, Basic{ 1, 0.5 }, true },
	// Inequalities
	DeepEqualTest{ 1, 2, false },
	DeepEqualTest{ int32(1), int32(2), false },
	DeepEqualTest{ 0.5, 0.6, false },
	DeepEqualTest{ float32(0.5), float32(0.6), false },
	DeepEqualTest{ "hello", "hey", false },
	DeepEqualTest{ make([]int, 10), make([]int, 11), false },
	DeepEqualTest{ &[3]int{ 1, 2, 3 }, &[3]int{ 1, 2, 4 }, false },
	DeepEqualTest{ Basic{ 1, 0.5 }, Basic{ 1, 0.6 }, false },
	// Mismatched types
	DeepEqualTest{ 1, 1.0, false },
	DeepEqualTest{ int32(1), int64(1), false },
	DeepEqualTest{ 0.5, "hello", false },
	DeepEqualTest{ []int{ 1, 2, 3 }, [3]int{ 1, 2, 3 }, false },
	DeepEqualTest{ &[3]interface{} { 1, 2, 4 }, &[3]interface{} { 1, 2, "s" }, false },
}

func TestDeepEqual(t *testing.T) {
	for i, test := range deepEqualTests {
		if r := DeepEqual(test.a, test.b); r != test.eq {
			t.Errorf("DeepEqual(%v, %v) = %v, want %v", test.a, test.b, r, test.eq);
		}
	}
}

func TestDeepEqualRecursiveStruct(t *testing.T) {
	a, b := new(Recursive), new(Recursive);
	*a = Recursive{ 12, a };
	*b = Recursive{ 12, b };
	if !DeepEqual(a, b) {
		t.Error("DeepEqual(recursive same) = false, want true");
	}
}

func TestDeepEqualComplexStruct(t *testing.T) {
	m := make(map[float]float);
	stra, strb := "hello", "hello";
	a, b := new(Complex), new(Complex);
	*a = Complex{5, [3]*Complex{a, b, a}, &stra, m};
	*b = Complex{5, [3]*Complex{b, a, a}, &strb, m};
	if !DeepEqual(a, b) {
		t.Error("DeepEqual(complex same) = false, want true");
	}
}

func TestDeepEqualComplexStructInequality(t *testing.T) {
	m := make(map[float]float);
	stra, strb := "hello", "helloo";  // Difference is here
	a, b := new(Complex), new(Complex);
	*a = Complex{5, [3]*Complex{a, b, a}, &stra, m};
	*b = Complex{5, [3]*Complex{b, a, a}, &strb, m};
	if DeepEqual(a, b) {
		t.Error("DeepEqual(complex different) = true, want false");
	}
}
