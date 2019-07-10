// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package log

import (
	"context"
	"fmt"
	"time"

	"golang.org/x/tools/internal/lsp/telemetry/tag"
)

type Entry struct {
	At      time.Time
	Message string
	Error   error
	Tags    tag.List
}

func ToEntry(ctx context.Context, at time.Time, tags tag.List) Entry {
	//TODO: filter more efficiently for the common case of stripping prefixes only
	entry := Entry{
		At: at,
	}
	for _, t := range tags {
		switch t.Key {
		case MessageTag:
			entry.Message = t.Value.(string)
		case ErrorTag:
			entry.Error = t.Value.(error)
		default:
			entry.Tags = append(entry.Tags, t)
		}
	}
	return entry
}

func (e Entry) Format(f fmt.State, r rune) {
	if !e.At.IsZero() {
		fmt.Fprint(f, e.At.Format("2006/01/02 15:04:05 "))
	}
	fmt.Fprint(f, e.Message)
	if e.Error != nil {
		fmt.Fprintf(f, ": %v", e.Error)
	}
	for _, tag := range e.Tags {
		fmt.Fprintf(f, "\n\t%v = %v", tag.Key, tag.Value)
	}
}
