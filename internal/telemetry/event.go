// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package telemetry

import (
	"fmt"
	"time"
)

type Event struct {
	At      time.Time
	Message string
	Error   error
	Tags    TagList
}

func (e Event) Format(f fmt.State, r rune) {
	if !e.At.IsZero() {
		fmt.Fprint(f, e.At.Format("2006/01/02 15:04:05 "))
	}
	fmt.Fprint(f, e.Message)
	if e.Error != nil {
		if f.Flag('+') {
			fmt.Fprintf(f, ": %+v", e.Error)
		} else {
			fmt.Fprintf(f, ": %v", e.Error)
		}
	}
	for _, tag := range e.Tags {
		fmt.Fprintf(f, "\n\t%v = %v", tag.Key, tag.Value)
	}
}
