// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package draw

import (
	"image"
)

// A Context represents a single graphics window.
type Context interface {
	// Screen returns an editable Image of window.
	Screen() Image

	// FlushImage flushes changes made to Screen() back to screen.
	FlushImage()

	// EventChan returns a channel carrying UI events such as key presses,
	// mouse movements and window resizes.
	EventChan() <-chan interface{}
}

// A KeyEvent is sent for a key press or release.
type KeyEvent struct {
	// The value k represents key k being pressed.
	// The value -k represents key k being released.
	// The specific set of key values is not specified,
	// but ordinary characters represent themselves.
	Key int
}

// A MouseEvent is sent for a button press or release or for a mouse movement.
type MouseEvent struct {
	// Buttons is a bit mask of buttons: 1<<0 is left, 1<<1 middle, 1<<2 right.
	// It represents button state and not necessarily the state delta: bit 0
	// being on means that the left mouse button is down, but does not imply
	// that the same button was up in the previous MouseEvent.
	Buttons int
	// Loc is the location of the cursor.
	Loc image.Point
	// Nsec is the event's timestamp.
	Nsec int64
}

// A ConfigEvent is sent each time the window's color model or size changes.
// The client should respond by calling Context.Screen to obtain a new image.
type ConfigEvent struct {
	Config image.Config
}
