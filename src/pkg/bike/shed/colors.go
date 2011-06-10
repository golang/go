// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package shed defines colors for bike sheds.

See http://red.bikeshed.org/ for more details.

TODO: More colors, colour support, stripes, methods, ponies.
*/
package shed

// A Color represents a color, or a colour if you're colonial enough.
type Color uint8

const (
	Red Color = iota
	Green
	Yellow
	Blue
	Purple
	Magenta
	Chartreuse
	Cyan
)
