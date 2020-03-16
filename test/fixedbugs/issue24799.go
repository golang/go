// run

// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Loads of 8 byte go.strings cannot use DS relocation
// in case the alignment is not a multiple of 4.

package main

import (
        "fmt"
)

type Level string

// The following are all go.strings. A link time error can
// occur if an 8 byte load is used to load a go.string that is
// not aligned to 4 bytes due to the type of relocation that
// is generated for the instruction. A fix was made to avoid
// generating an instruction with DS relocation for go.strings
// since their alignment is not known until link time. 

// This problem only affects go.string since other types have
// correct alignment.

const (
        LevelBad Level = "badvals"
        LevelNone Level = "No"
        LevelMetadata Level = "Metadata"
        LevelRequest Level = "Request"
        LevelRequestResponse Level = "RequestResponse"
)

func ordLevel(l Level) int {
        switch l {
        case LevelMetadata:
                return 1
        case LevelRequest:
                return 2
        case LevelRequestResponse:
                return 3
        default:
                return 0
        }
}

//go:noinline
func test(l Level) {
        if ordLevel(l) < ordLevel(LevelMetadata) {
                fmt.Printf("OK\n")
        }
}

func main() {
        test(LevelMetadata)
}
