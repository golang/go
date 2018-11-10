// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package race_test

import (
	"reflect"
	"testing"
)

func TestRaceReflectRW(t *testing.T) {
	ch := make(chan bool, 1)
	i := 0
	v := reflect.ValueOf(&i)
	go func() {
		v.Elem().Set(reflect.ValueOf(1))
		ch <- true
	}()
	_ = v.Elem().Int()
	<-ch
}

func TestRaceReflectWW(t *testing.T) {
	ch := make(chan bool, 1)
	i := 0
	v := reflect.ValueOf(&i)
	go func() {
		v.Elem().Set(reflect.ValueOf(1))
		ch <- true
	}()
	v.Elem().Set(reflect.ValueOf(2))
	<-ch
}

func TestRaceReflectCopyWW(t *testing.T) {
	ch := make(chan bool, 1)
	a := make([]byte, 2)
	v := reflect.ValueOf(a)
	go func() {
		reflect.Copy(v, v)
		ch <- true
	}()
	reflect.Copy(v, v)
	<-ch
}
