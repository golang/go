package main

import (
	"iter"
	"reflect"
)

type T int

func (T) M() { println("M called") }

func main() {
	v := reflect.ValueOf(T(0))
	t := v.Type()
	tv := reflect.ValueOf(t)
	ms := tv.MethodByName("Methods")
	it := ms.Call(nil)[0]
	for m := range it.Interface().(iter.Seq[reflect.Method]) {
		m.Func.Call([]reflect.Value{v})
	}
}
