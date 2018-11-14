// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake1

// This is a test file for the behaviors in Exported.Expect.

type AThing string //@AThing,mark(StringThing, "AThing"),mark(REThing,re`.T.*g`)

type Match string //@check("Match",re`[[:upper:]]`)

//@check(AThing, StringThing)
//@check(AThing, REThing)

//@boolArg(true, false)
//@intArg(42)
//@stringArg(PlainString, "PlainString")
//@stringArg(IdentAsString,IdentAsString)
//@directNote()
//@range(AThing)

// The following test should remain at the bottom of the file
//@checkEOF(EOF)
