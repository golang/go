// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"fmt"
	"log"
	"reflect"
)

func Decode() {
	b := []byte(`{"Name":"Wednesday","Age":6,"Parents":["Gomez","Morticia"]}`)

	var f interface{}
	err := json.Unmarshal(b, &f)

	// STOP OMIT

	if err != nil {
		panic(err)
	}

	expected := map[string]interface{}{
		"Name": "Wednesday",
		"Age":  float64(6),
		"Parents": []interface{}{
			"Gomez",
			"Morticia",
		},
	}

	if !reflect.DeepEqual(f, expected) {
		log.Panicf("Error unmarshalling %q, expected %q, got %q", b, expected, f)
	}

	f = map[string]interface{}{
		"Name": "Wednesday",
		"Age":  6,
		"Parents": []interface{}{
			"Gomez",
			"Morticia",
		},
	}

	// STOP OMIT

	m := f.(map[string]interface{})

	for k, v := range m {
		switch vv := v.(type) {
		case string:
			fmt.Println(k, "is string", vv)
		case int:
			fmt.Println(k, "is int", vv)
		case []interface{}:
			fmt.Println(k, "is an array:")
			for i, u := range vv {
				fmt.Println(i, u)
			}
		default:
			fmt.Println(k, "is of a type I don't know how to handle")
		}
	}

	// STOP OMIT
}

func main() {
	Decode()
}
