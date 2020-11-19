package main

import (
	"encoding/json"
	"fmt"
)

func main() { // function
	fmt.Println("Hello")
}

var myvar int // variable

type myType string // basic type

type myDecoder json.Decoder // to use the encoding/json import

func (m *myType) Blahblah() {} // method

type myStruct struct { // struct type
	myStructField int // struct field
}

type myInterface interface { // interface
	DoSomeCoolStuff() string // interface method
}

type embed struct {
	myStruct

	nestedStruct struct {
		nestedField int

		nestedStruct2 struct {
			int
		}
	}

	nestedInterface interface {
		myInterface
		nestedMethod()
	}
}

func Dunk() int { return 0 }

func dunk() {}
