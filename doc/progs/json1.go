// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"log"
	"reflect"
)

type Message struct {
	Name string
	Body string
	Time int64
}

// STOP OMIT

func Encode() {
	m := Message{"Alice", "Hello", 1294706395881547000}
	b, err := json.Marshal(m)

	if err != nil {
		panic(err)
	}

	expected := []byte(`{"Name":"Alice","Body":"Hello","Time":1294706395881547000}`)
	if !reflect.DeepEqual(b, expected) {
		log.Panicf("Error marshalling %q, expected %q, got %q.", m, expected, b)
	}

}

func Decode() {
	b := []byte(`{"Name":"Alice","Body":"Hello","Time":1294706395881547000}`)
	var m Message
	err := json.Unmarshal(b, &m)

	if err != nil {
		panic(err)
	}

	expected := Message{
		Name: "Alice",
		Body: "Hello",
		Time: 1294706395881547000,
	}

	if !reflect.DeepEqual(m, expected) {
		log.Panicf("Error unmarshalling %q, expected %q, got %q.", b, expected, m)
	}

	m = Message{
		Name: "Alice",
		Body: "Hello",
		Time: 1294706395881547000,
	}

	// STOP OMIT
}

func PartialDecode() {
	b := []byte(`{"Name":"Bob","Food":"Pickle"}`)
	var m Message
	err := json.Unmarshal(b, &m)

	// STOP OMIT

	if err != nil {
		panic(err)
	}

	expected := Message{
		Name: "Bob",
	}

	if !reflect.DeepEqual(expected, m) {
		log.Panicf("Error unmarshalling %q, expected %q, got %q.", b, expected, m)
	}
}

func main() {
	Encode()
	Decode()
	PartialDecode()
}
