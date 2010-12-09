// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package inotify

import (
	"os"
	"time"
	"testing"
)

func TestInotifyEvents(t *testing.T) {
	// Create an inotify watcher instance and initialize it
	watcher, err := NewWatcher()
	if err != nil {
		t.Fatalf("NewWatcher() failed: %s", err)
	}

	// Add a watch for "_obj"
	err = watcher.Watch("_obj")
	if err != nil {
		t.Fatalf("Watcher.Watch() failed: %s", err)
	}

	// Receive errors on the error channel on a separate goroutine
	go func() {
		for err := range watcher.Error {
			t.Fatalf("error received: %s", err)
		}
	}()

	const testFile string = "_obj/TestInotifyEvents.testfile"

	// Receive events on the event channel on a separate goroutine
	eventstream := watcher.Event
	var eventsReceived = 0
	go func() {
		for event := range eventstream {
			// Only count relevant events
			if event.Name == testFile {
				eventsReceived++
				t.Logf("event received: %s", event)
			} else {
				t.Logf("unexpected event received: %s", event)
			}
		}
	}()

	// Create a file
	// This should add at least one event to the inotify event queue
	_, err = os.Open(testFile, os.O_WRONLY|os.O_CREAT, 0666)
	if err != nil {
		t.Fatalf("creating test file failed: %s", err)
	}

	// We expect this event to be received almost immediately, but let's wait 1 s to be sure
	time.Sleep(1000e6) // 1000 ms
	if eventsReceived == 0 {
		t.Fatal("inotify event hasn't been received after 1 second")
	}

	// Try closing the inotify instance
	t.Log("calling Close()")
	watcher.Close()
	t.Log("waiting for the event channel to become closed...")
	var i = 0
	for !closed(eventstream) {
		if i >= 20 {
			t.Fatal("event stream was not closed after 1 second, as expected")
		}
		t.Log("waiting for 50 ms...")
		time.Sleep(50e6) // 50 ms
		i++
	}
	t.Log("event channel closed")
}


func TestInotifyClose(t *testing.T) {
	watcher, _ := NewWatcher()
	watcher.Close()

	done := false
	go func() {
		watcher.Close()
		done = true
	}()

	time.Sleep(50e6) // 50 ms
	if !done {
		t.Fatal("double Close() test failed: second Close() call didn't return")
	}

	err := watcher.Watch("_obj")
	if err == nil {
		t.Fatal("expected error on Watch() after Close(), got nil")
	}
}
