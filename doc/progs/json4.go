// run

// Copyright 2012 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package main

import (
	"encoding/json"
	"log"
	"reflect"
)

type FamilyMember struct {
	Name    string
	Age     int
	Parents []string
}

// STOP OMIT

func Decode() {
	b := []byte(`{"Name":"Bob","Age":20,"Parents":["Morticia", "Gomez"]}`)
	var m FamilyMember
	err := json.Unmarshal(b, &m)

	// STOP OMIT

	if err != nil {
		panic(err)
	}

	expected := FamilyMember{
		Name:    "Bob",
		Age:     20,
		Parents: []string{"Morticia", "Gomez"},
	}

	if !reflect.DeepEqual(expected, m) {
		log.Panicf("Error unmarshalling %q, expected %q, got %q", b, expected, m)
	}
}

func main() {
	Decode()
}
