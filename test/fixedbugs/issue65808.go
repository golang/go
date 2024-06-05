// compile

// Copyright 2024 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.package main

package main

type Stringer interface {
	String() string
}

type (
	stringer  struct{}
	stringers [2]stringer
	foo       struct {
		stringers
	}
)

func (stringer) String() string  { return "" }
func toString(s Stringer) string { return s.String() }

func (v stringers) toStrings() []string {
	return []string{toString(v[0]), toString(v[1])}
}

func main() {
	_ = stringers{}
}
