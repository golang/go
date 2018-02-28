package main

import (
	"reflect"
)

type Foo struct {
	Bar string `json:"Bar@baz,omitempty"`
}

func F() {
	println(reflect.TypeOf(Foo{}).Field(0).Tag)
}

func main() {}
