// Copyright 2022 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fallthroughs

func _() {
	var x int
	switch x {
	case 0:
		fallthrough

	case 1:
		fallthrough // ERROR fallthrough statement out of place
		{
		}

	case 2:
		{
			fallthrough // ERROR fallthrough statement out of place
		}

	case 3:
		for {
			fallthrough // ERROR fallthrough statement out of place
		}

	case 4:
		fallthrough // trailing empty statements are ok
		;
		;

	case 5:
		fallthrough

	default:
		fallthrough // ERROR cannot fallthrough final case in switch
	}

	fallthrough // ERROR fallthrough statement out of place

	if true {
		fallthrough // ERROR fallthrough statement out of place
	}

	for {
		fallthrough // ERROR fallthrough statement out of place
	}

	var t any
	switch t.(type) {
	case int:
		fallthrough // ERROR cannot fallthrough in type switch
	}
}
