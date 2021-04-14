// Copyright 2017 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package dynamodbstreamsevt

import "encoding/json"

var foo json.RawMessage

type Event struct{}

func (e *Event) Dummy() {}
