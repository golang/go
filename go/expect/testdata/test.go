// Copyright 2018 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package fake1

// The greek letters in this file mark points we use for marker tests.
// We use unique markers so we can make the tests stable against changes to
// this file.

const (
	_                   int = iota
	αSimpleMarkerα          //@αSimpleMarker
	offsetββMarker          //@mark(OffsetMarker, "β")
	regexγMaγrker           //@mark(RegexMarker, re`\p{Greek}Ma`)
	εMultipleεζMarkersζ     //@εMultiple,ζMarkers
	ηBlockMarkerη           /*@ηBlockMarker*/
)

/*Marker ι inside ι a comment*/ //@mark(Comment,"ι inside ")

func someFunc(a, b int) int {
	// The line below must be the first occurrence of the plus operator
	return a + b + 1 //@mark(NonIdentifier, re`\+[^\+]*`)
}

// And some extra checks for interesting action parameters
//@check(αSimpleMarker)
//@check(StringAndInt, "Number %d", 12)
//@check(Bool, true)
