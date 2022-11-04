// Copyright 2016 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Input for TestIssue13566

package a

import "encoding/json"

type A struct {
	a    *A
	json json.RawMessage
}
