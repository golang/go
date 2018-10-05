// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the unmarshal checker.

package testdata

import (
	"bytes"
	"encoding/gob"
	"encoding/json"
	"encoding/xml"
	"errors"
	"fmt"
)

func _() {
	type t struct {
		a int
	}
	var v t
	var r io.Reader

	json.Unmarshal([]byte{}, v) // ERROR "call of Unmarshal passes non-pointer as second argument"
	json.Unmarshal([]byte{}, &v)
	json.NewDecoder(r).Decode(v) // ERROR "call of Decode passes non-pointer"
	json.NewDecoder(r).Decode(&v)
	gob.NewDecoder(r).Decode(v) // ERROR "call of Decode passes non-pointer"
	gob.NewDecoder(r).Decode(&v)
	xml.Unmarshal([]byte{}, v) // ERROR "call of Unmarshal passes non-pointer as second argument"
	xml.Unmarshal([]byte{}, &v)
	xml.NewDecoder(r).Decode(v) // ERROR "call of Decode passes non-pointer"
	xml.NewDecoder(r).Decode(&v)

	var p *t
	json.Unmarshal([]byte{}, p)
	json.Unmarshal([]byte{}, *p) // ERROR "call of Unmarshal passes non-pointer as second argument"
	json.NewDecoder(r).Decode(p)
	json.NewDecoder(r).Decode(*p) // ERROR "call of Decode passes non-pointer"
	gob.NewDecoder(r).Decode(p)
	gob.NewDecoder(r).Decode(*p) // ERROR "call of Decode passes non-pointer"
	xml.Unmarshal([]byte{}, p)
	xml.Unmarshal([]byte{}, *p) // ERROR "call of Unmarshal passes non-pointer as second argument"
	xml.NewDecoder(r).Decode(p)
	xml.NewDecoder(r).Decode(*p) // ERROR "call of Decode passes non-pointer"

	var i interface{}
	json.Unmarshal([]byte{}, i)
	json.NewDecoder(r).Decode(i)

	json.Unmarshal([]byte{}, nil)               // ERROR "call of Unmarshal passes non-pointer as second argument"
	json.Unmarshal([]byte{}, []t{})             // ERROR "call of Unmarshal passes non-pointer as second argument"
	json.Unmarshal([]byte{}, map[string]int{})  // ERROR "call of Unmarshal passes non-pointer as second argument"
	json.NewDecoder(r).Decode(nil)              // ERROR "call of Decode passes non-pointer"
	json.NewDecoder(r).Decode([]t{})            // ERROR "call of Decode passes non-pointer"
	json.NewDecoder(r).Decode(map[string]int{}) // ERROR "call of Decode passes non-pointer"

	json.Unmarshal(func() ([]byte, interface{}) { return []byte{}, v }())
}
