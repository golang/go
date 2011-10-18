// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

/*
Package inotify implements a wrapper for the Linux inotify system.

Example:
    watcher, err := inotify.NewWatcher()
    if err != nil {
        log.Fatal(err)
    }
    err = watcher.Watch("/tmp")
    if err != nil {
        log.Fatal(err)
    }
    for {
        select {
        case ev := <-watcher.Event:
            log.Println("event:", ev)
        case err := <-watcher.Error:
            log.Println("error:", err)
        }
    }

*/
package inotify

import (
	"fmt"
	"os"
	"strings"
	"syscall"
	"unsafe"
)

type Event struct {
	Mask   uint32 // Mask of events
	Cookie uint32 // Unique cookie associating related events (for rename(2))
	Name   string // File name (optional)
}

type watch struct {
	wd    uint32 // Watch descriptor (as returned by the inotify_add_watch() syscall)
	flags uint32 // inotify flags of this watch (see inotify(7) for the list of valid flags)
}

type Watcher struct {
	fd       int               // File descriptor (as returned by the inotify_init() syscall)
	watches  map[string]*watch // Map of inotify watches (key: path)
	paths    map[int]string    // Map of watched paths (key: watch descriptor)
	Error    chan os.Error     // Errors are sent on this channel
	Event    chan *Event       // Events are returned on this channel
	done     chan bool         // Channel for sending a "quit message" to the reader goroutine
	isClosed bool              // Set to true when Close() is first called
}

// NewWatcher creates and returns a new inotify instance using inotify_init(2)
func NewWatcher() (*Watcher, os.Error) {
	fd, errno := syscall.InotifyInit()
	if fd == -1 {
		return nil, os.NewSyscallError("inotify_init", errno)
	}
	w := &Watcher{
		fd:      fd,
		watches: make(map[string]*watch),
		paths:   make(map[int]string),
		Event:   make(chan *Event),
		Error:   make(chan os.Error),
		done:    make(chan bool, 1),
	}

	go w.readEvents()
	return w, nil
}

// Close closes an inotify watcher instance
// It sends a message to the reader goroutine to quit and removes all watches
// associated with the inotify instance
func (w *Watcher) Close() os.Error {
	if w.isClosed {
		return nil
	}
	w.isClosed = true

	// Send "quit" message to the reader goroutine
	w.done <- true
	for path := range w.watches {
		w.RemoveWatch(path)
	}

	return nil
}

// AddWatch adds path to the watched file set.
// The flags are interpreted as described in inotify_add_watch(2).
func (w *Watcher) AddWatch(path string, flags uint32) os.Error {
	if w.isClosed {
		return os.NewError("inotify instance already closed")
	}

	watchEntry, found := w.watches[path]
	if found {
		watchEntry.flags |= flags
		flags |= syscall.IN_MASK_ADD
	}
	wd, errno := syscall.InotifyAddWatch(w.fd, path, flags)
	if wd == -1 {
		return &os.PathError{"inotify_add_watch", path, os.Errno(errno)}
	}

	if !found {
		w.watches[path] = &watch{wd: uint32(wd), flags: flags}
		w.paths[wd] = path
	}
	return nil
}

// Watch adds path to the watched file set, watching all events.
func (w *Watcher) Watch(path string) os.Error {
	return w.AddWatch(path, IN_ALL_EVENTS)
}

// RemoveWatch removes path from the watched file set.
func (w *Watcher) RemoveWatch(path string) os.Error {
	watch, ok := w.watches[path]
	if !ok {
		return os.NewError(fmt.Sprintf("can't remove non-existent inotify watch for: %s", path))
	}
	success, errno := syscall.InotifyRmWatch(w.fd, watch.wd)
	if success == -1 {
		return os.NewSyscallError("inotify_rm_watch", errno)
	}
	delete(w.watches, path)
	return nil
}

// readEvents reads from the inotify file descriptor, converts the
// received events into Event objects and sends them via the Event channel
func (w *Watcher) readEvents() {
	var (
		buf   [syscall.SizeofInotifyEvent * 4096]byte // Buffer for a maximum of 4096 raw events
		n     int                                     // Number of bytes read with read()
		errno int                                     // Syscall errno
	)

	for {
		n, errno = syscall.Read(w.fd, buf[0:])
		// See if there is a message on the "done" channel
		var done bool
		select {
		case done = <-w.done:
		default:
		}

		// If EOF or a "done" message is received
		if n == 0 || done {
			errno := syscall.Close(w.fd)
			if errno == -1 {
				w.Error <- os.NewSyscallError("close", errno)
			}
			close(w.Event)
			close(w.Error)
			return
		}
		if n < 0 {
			w.Error <- os.NewSyscallError("read", errno)
			continue
		}
		if n < syscall.SizeofInotifyEvent {
			w.Error <- os.NewError("inotify: short read in readEvents()")
			continue
		}

		var offset uint32 = 0
		// We don't know how many events we just read into the buffer
		// While the offset points to at least one whole event...
		for offset <= uint32(n-syscall.SizeofInotifyEvent) {
			// Point "raw" to the event in the buffer
			raw := (*syscall.InotifyEvent)(unsafe.Pointer(&buf[offset]))
			event := new(Event)
			event.Mask = uint32(raw.Mask)
			event.Cookie = uint32(raw.Cookie)
			nameLen := uint32(raw.Len)
			// If the event happened to the watched directory or the watched file, the kernel
			// doesn't append the filename to the event, but we would like to always fill the
			// the "Name" field with a valid filename. We retrieve the path of the watch from
			// the "paths" map.
			event.Name = w.paths[int(raw.Wd)]
			if nameLen > 0 {
				// Point "bytes" at the first byte of the filename
				bytes := (*[syscall.PathMax]byte)(unsafe.Pointer(&buf[offset+syscall.SizeofInotifyEvent]))
				// The filename is padded with NUL bytes. TrimRight() gets rid of those.
				event.Name += "/" + strings.TrimRight(string(bytes[0:nameLen]), "\000")
			}
			// Send the event on the events channel
			w.Event <- event

			// Move to the next event in the buffer
			offset += syscall.SizeofInotifyEvent + nameLen
		}
	}
}

// String formats the event e in the form
// "filename: 0xEventMask = IN_ACCESS|IN_ATTRIB_|..."
func (e *Event) String() string {
	var events string = ""

	m := e.Mask
	for _, b := range eventBits {
		if m&b.Value != 0 {
			m &^= b.Value
			events += "|" + b.Name
		}
	}

	if m != 0 {
		events += fmt.Sprintf("|%#x", m)
	}
	if len(events) > 0 {
		events = " == " + events[1:]
	}

	return fmt.Sprintf("%q: %#x%s", e.Name, e.Mask, events)
}

const (
	// Options for inotify_init() are not exported
	// IN_CLOEXEC    uint32 = syscall.IN_CLOEXEC
	// IN_NONBLOCK   uint32 = syscall.IN_NONBLOCK

	// Options for AddWatch
	IN_DONT_FOLLOW uint32 = syscall.IN_DONT_FOLLOW
	IN_ONESHOT     uint32 = syscall.IN_ONESHOT
	IN_ONLYDIR     uint32 = syscall.IN_ONLYDIR

	// The "IN_MASK_ADD" option is not exported, as AddWatch
	// adds it automatically, if there is already a watch for the given path
	// IN_MASK_ADD      uint32 = syscall.IN_MASK_ADD

	// Events
	IN_ACCESS        uint32 = syscall.IN_ACCESS
	IN_ALL_EVENTS    uint32 = syscall.IN_ALL_EVENTS
	IN_ATTRIB        uint32 = syscall.IN_ATTRIB
	IN_CLOSE         uint32 = syscall.IN_CLOSE
	IN_CLOSE_NOWRITE uint32 = syscall.IN_CLOSE_NOWRITE
	IN_CLOSE_WRITE   uint32 = syscall.IN_CLOSE_WRITE
	IN_CREATE        uint32 = syscall.IN_CREATE
	IN_DELETE        uint32 = syscall.IN_DELETE
	IN_DELETE_SELF   uint32 = syscall.IN_DELETE_SELF
	IN_MODIFY        uint32 = syscall.IN_MODIFY
	IN_MOVE          uint32 = syscall.IN_MOVE
	IN_MOVED_FROM    uint32 = syscall.IN_MOVED_FROM
	IN_MOVED_TO      uint32 = syscall.IN_MOVED_TO
	IN_MOVE_SELF     uint32 = syscall.IN_MOVE_SELF
	IN_OPEN          uint32 = syscall.IN_OPEN

	// Special events
	IN_ISDIR      uint32 = syscall.IN_ISDIR
	IN_IGNORED    uint32 = syscall.IN_IGNORED
	IN_Q_OVERFLOW uint32 = syscall.IN_Q_OVERFLOW
	IN_UNMOUNT    uint32 = syscall.IN_UNMOUNT
)

var eventBits = []struct {
	Value uint32
	Name  string
}{
	{IN_ACCESS, "IN_ACCESS"},
	{IN_ATTRIB, "IN_ATTRIB"},
	{IN_CLOSE, "IN_CLOSE"},
	{IN_CLOSE_NOWRITE, "IN_CLOSE_NOWRITE"},
	{IN_CLOSE_WRITE, "IN_CLOSE_WRITE"},
	{IN_CREATE, "IN_CREATE"},
	{IN_DELETE, "IN_DELETE"},
	{IN_DELETE_SELF, "IN_DELETE_SELF"},
	{IN_MODIFY, "IN_MODIFY"},
	{IN_MOVE, "IN_MOVE"},
	{IN_MOVED_FROM, "IN_MOVED_FROM"},
	{IN_MOVED_TO, "IN_MOVED_TO"},
	{IN_MOVE_SELF, "IN_MOVE_SELF"},
	{IN_OPEN, "IN_OPEN"},
	{IN_ISDIR, "IN_ISDIR"},
	{IN_IGNORED, "IN_IGNORED"},
	{IN_Q_OVERFLOW, "IN_Q_OVERFLOW"},
	{IN_UNMOUNT, "IN_UNMOUNT"},
}
