// Copyright 2010 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// +build linux

package inotify

import (
	"os"
	"testing"
	"time"
)

func TestInotifyEvents(t *testing.T) {
	// Create an inotify watcher instance and initialize it
	watcher, err := NewWatcher()
	if err != nil {
		t.Fatalf("NewWatcher() failed: %s", err)
	}

	t.Logf("NEEDS TO BE CONVERTED TO NEW GO TOOL") // TODO
	return

	// Add a watch for "_test"
	err = watcher.Watch("_test")
	if err != nil {
		t.Fatalf("Watcher.Watch() failed: %s", err)
	}

	// Receive errors on the error channel on a separate goroutine
	go func() {
		for err := range watcher.Error {
			t.Fatalf("error received: %s", err)
		}
	}()

	const testFile string = "_test/TestInotifyEvents.testfile"

	// Receive events on the event channel on a separate goroutine
	eventstream := watcher.Event
	var eventsReceived = 0
	done := make(chan bool)
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
		done <- true
	}()

	// Create a file
	// This should add at least one event to the inotify event queue
	_, err = os.OpenFile(testFile, os.O_WRONLY|os.O_CREATE, 0666)
	if err != nil {
		t.Fatalf("creating test file failed: %s", err)
	}

	// We expect this event to be received almost immediately, but let's wait 1 s to be sure
	time.Sleep(1 * time.Second)
	if eventsReceived == 0 {
		t.Fatal("inotify event hasn't been received after 1 second")
	}

	// Try closing the inotify instance
	t.Log("calling Close()")
	watcher.Close()
	t.Log("waiting for the event channel to become closed...")
	select {
	case <-done:
		t.Log("event channel closed")
	case <-time.After(1 * time.Second):
		t.Fatal("event stream was not closed after 1 second")
	}
}

func TestInotifyClose(t *testing.T) {
	watcher, _ := NewWatcher()
	watcher.Close()

	done := false
	go func() {
		watcher.Close()
		done = true
	}()

	time.Sleep(50 * time.Millisecond)
	if !done {
		t.Fatal("double Close() test failed: second Close() call didn't return")
	}

	err := watcher.Watch("_test")
	if err == nil {
		t.Fatal("expected error on Watch() after Close(), got nil")
	}
}
