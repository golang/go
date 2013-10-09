// +build ignore

package main

import "reflect"

var zero, a, b int

// func f(p *int) *int {
// 	print(p) // #@pointsto
// 	return &b
// }

// func g(p *bool) {
// }

// func reflectValueCall() {
// 	rvf := reflect.ValueOf(f)
// 	res := rvf.Call([]reflect.Value{reflect.ValueOf(&a)})
// 	print(res[0].Interface())        // #@types
// 	print(res[0].Interface().(*int)) // #@pointsto
// }

// #@calls main.reflectValueCall -> main.f

func reflectTypeInOut() {
	var f func(float64, bool) (string, int)
	// TODO(adonovan): when the In/Out argument is a valid index constant,
	// only include a single type in the result.  Needs some work.
	print(reflect.Zero(reflect.TypeOf(f).In(0)).Interface())    // @types float64 | bool
	print(reflect.Zero(reflect.TypeOf(f).In(1)).Interface())    // @types float64 | bool
	print(reflect.Zero(reflect.TypeOf(f).In(-1)).Interface())   // @types float64 | bool
	print(reflect.Zero(reflect.TypeOf(f).In(zero)).Interface()) // @types float64 | bool

	print(reflect.Zero(reflect.TypeOf(f).Out(0)).Interface()) // @types string | int
	print(reflect.Zero(reflect.TypeOf(f).Out(1)).Interface()) // @types string | int
	print(reflect.Zero(reflect.TypeOf(f).Out(2)).Interface()) // @types string | int
	print(reflect.Zero(reflect.TypeOf(3).Out(0)).Interface()) // @types
}

type T struct{}

func (T) F()    {}
func (T) g(int) {}

type U struct{}

func (U) F(int)    {}
func (U) g(string) {}

var nonconst string

func reflectTypeMethodByName() {
	TU := reflect.TypeOf([]interface{}{T{}, U{}}[0])
	print(reflect.Zero(TU)) // @types T | U

	F, _ := TU.MethodByName("F")
	print(reflect.Zero(F.Type)) // @types func(T) | func(U, int)
	print(F.Func)               // @pointsto (main.T).F | (main.U).F

	g, _ := TU.MethodByName("g")
	print(reflect.Zero(g.Type)) // @types func(T, int) | func(U, string)
	print(g.Func)               // @pointsto (main.T).g | (main.U).g

	// Non-literal method names are treated less precisely.
	U := reflect.TypeOf(U{})
	X, _ := U.MethodByName(nonconst)
	print(reflect.Zero(X.Type)) // @types func(U, int) | func(U, string)
	print(X.Func)               // @pointsto (main.U).F | (main.U).g
}

func reflectTypeMethod() {
	m := reflect.TypeOf(T{}).Method(0)
	print(reflect.Zero(m.Type)) // @types func(T) | func(T, int)
	print(m.Func)               // @pointsto (main.T).F | (main.T).g
}

func main() {
	//reflectValueCall()
	reflectTypeInOut()
	reflectTypeMethodByName()
	reflectTypeMethod()
}
