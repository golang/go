// Copyright 2023 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Represents APIs to compare JSON strings

package json

import (
	"reflect"
)

type jsonType interface {
	[]byte | string
}

// Eq reports whether the jsonStrings are “equal” defined as follows.
// Keys are same but can be of different order
// Values: should should
//
// Caveat: arrays: order should be matched
func Eq[T jsonType](s1, s2 T) (bool, error) {
	var i, i2 any
	err := Unmarshal([]byte(s1), &i)
	if err != nil {
		return false, err
	}
	err = Unmarshal([]byte(s2), &i2)
	if err != nil {
		return false, err
	}
	return reflect.DeepEqual(i, i2), nil
}

// DeeplyEqual reports whether the jsonStrings are “equal” defined as follows.
// Keys are same but can be of different order
// Values: should should
//
// Caveat: arrays: order should be matched
func DeeplyEqual[T jsonType](s1, s2 T, sN ...T) (eq bool) {
	eq, _ = Equal(s1, s2, sN...)
	return
}

// Equal reports whether the jsonStrings are “equal” defined as follows.
// Keys are same but can be of different order
// Values: should should
//
// Caveat: arrays: order should be matched
func Equal[T jsonType](s1, s2 T, sN ...T) (eq bool, err error) {
	if len(sN) == 0 {
		eq, err = Eq(s1, s2)
		return
	}
	sN = append(sN, s2)

	for _, js := range sN {
		eq, err = Eq(s1, js)
		if !eq {
			return
		}

	}
	return
}
