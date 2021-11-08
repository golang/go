// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package a

import (
	"errors"
	"strings"
)

var G interface{}

func Unmarshal(data []byte, o interface{}) error {
	G = o
	v, ok := o.(*map[string]interface{})
	if !ok {
		return errors.New("eek")
	}
	vals := make(map[string]interface{})
	s := string(data)
	items := strings.Split(s, " ")
	var err error
	for _, item := range items {
		vals[item] = s
		if item == "error" {
			err = errors.New("ouch")
		}
	}
	*v = vals
	return err
}
