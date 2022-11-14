// Copyright 2020 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package p

func _[T any](x interface{}){
	switch x.(type) {
	case T: // ok to use a type parameter
	case int:
	}

	switch x.(type) {
	case T:
	case T /* ERROR duplicate case */ :
	}
}

type constraint interface {
	~int
}

func _[T constraint](x interface{}){
	switch x.(type) {
	case T: // ok to use a type parameter even if type set contains int
	case int:
	}
}

func _(x constraint /* ERROR contains type constraints */ ) {
	switch x.(type) { // no need to report another error
	}
}
