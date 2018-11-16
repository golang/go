// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build gofuzz

package traceparser

import (
	"bytes"
	"fmt"
	"log"
)

// at first we ran the old parser, and return 0 if it failed, on the theory that we don't have
// to do better. But that leads to very few crashes to look at.
// Maybe better just to make it so that the new parser doesn't misbehave, and if it doesn't get
// an error, that the old parser gets the same results. (up to whatever)
// perhaps even better would be to seed corpus with examples from which the 16-byte header
// has been stripped, and add it in Fuzz, so the fuzzer doesn't spend a lot of time making
// changes we reject in the header. (this may not be necessary)

func Fuzz(data []byte) int {
	if len(data) < 16 {
		return 0
	}
	switch x := string(data[:16]); x {
	default:
		return 0
	case "go 1.9 trace\000\000\000\000":
		break
	case "go 1.10 trace\000\000\000":
		break
	case "go 1.11 trace\000\000\000":
		break
	}
	p, errp := ParseBuffer(bytes.NewBuffer(data))
	if errp != nil {
		if p != nil {
			panic(fmt.Sprintf("p not nil on error %s", errp))
		}
	}
	// TODO(pjw): if no errors, compare parses?
	return 1
}

func init() {
	log.SetFlags(log.Lshortfile)
}
