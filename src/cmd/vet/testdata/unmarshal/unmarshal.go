// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// This file contains tests for the unmarshal checker.

package unmarshal

import "encoding/json"

func _() {
	type t struct {
		a int
	}
	var v t

	json.Unmarshal([]byte{}, v) // ERROR "call of Unmarshal passes non-pointer as second argument"
}
