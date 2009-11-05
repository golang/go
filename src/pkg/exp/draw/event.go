// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package draw

// A Context represents a single graphics window.
type Context interface {
	// Screen returns an editable Image of window.
	Screen() Image;

	// FlushImage flushes changes made to Screen() back to screen.
	FlushImage();

	// KeyboardChan returns a channel carrying keystrokes.
	// An event is sent each time a key is pressed or released.
	// The value k represents key k being pressed.
	// The value -k represents key k being released.
	// The specific set of key values is not specified,
	// but ordinary character represent themselves.
	KeyboardChan() <-chan int;

	// MouseChan returns a channel carrying mouse events.
	// A new event is sent each time the mouse moves or a
	// button is pressed or released.
	MouseChan() <-chan Mouse;

	// ResizeChan returns a channel carrying resize events.
	// An event is sent each time the window is resized;
	// the client should respond by calling Screen() to obtain
	// the new screen image.
	// The value sent on the channel is always ``true'' and can be ignored.
	ResizeChan() <-chan bool;

	// QuitChan returns a channel carrying quit requests.
	// After reading a value from the quit channel, the application
	// should exit.
	QuitChan() <-chan bool;
}

// A Mouse represents the state of the mouse.
type Mouse struct {
	Buttons	int;	// bit mask of buttons: 1<<0 is left, 1<<1 middle, 1<<2 right
	Point;		// location of cursor
	Nsec	int64;	// time stamp
}
