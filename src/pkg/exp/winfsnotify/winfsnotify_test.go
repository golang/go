// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package winfsnotify

import (
	"os"
	"testing"
	"time"
)

func expect(t *testing.T, eventstream <-chan *Event, name string, mask uint32) {
	t.Logf(`expected: "%s": 0x%x`, name, mask)
	select {
	case event := <-eventstream:
		if event == nil {
			t.Fatal("nil event received")
		}
		t.Logf("received: %s", event)
		if event.Name != name || event.Mask != mask {
			t.Fatal("did not receive expected event")
		}
	case <-time.After(1 * time.Second):
		t.Fatal("timed out waiting for event")
	}
}

func TestNotifyEvents(t *testing.T) {
	watcher, err := NewWatcher()
	if err != nil {
		t.Fatalf("NewWatcher() failed: %s", err)
	}

	testDir := "TestNotifyEvents.testdirectory"
	testFile := testDir + "/TestNotifyEvents.testfile"
	testFile2 := testFile + ".new"
	const mask = FS_ALL_EVENTS & ^(FS_ATTRIB|FS_CLOSE) | FS_IGNORED

	// Add a watch for testDir
	os.RemoveAll(testDir)
	if err = os.Mkdir(testDir, 0777); err != nil {
		t.Fatalf("Failed to create test directory: %s", err)
	}
	defer os.RemoveAll(testDir)
	err = watcher.AddWatch(testDir, mask)
	if err != nil {
		t.Fatalf("Watcher.Watch() failed: %s", err)
	}

	// Receive errors on the error channel on a separate goroutine
	go func() {
		for err := range watcher.Error {
			t.Fatalf("error received: %s", err)
		}
	}()

	// Create a file
	file, err := os.Create(testFile)
	if err != nil {
		t.Fatalf("creating test file failed: %s", err)
	}
	expect(t, watcher.Event, testFile, FS_CREATE)

	err = watcher.AddWatch(testFile, mask)
	if err != nil {
		t.Fatalf("Watcher.Watch() failed: %s", err)
	}

	if _, err = file.WriteString("hello, world"); err != nil {
		t.Fatalf("failed to write to test file: %s", err)
	}
	if err = file.Close(); err != nil {
		t.Fatalf("failed to close test file: %s", err)
	}
	expect(t, watcher.Event, testFile, FS_MODIFY)
	expect(t, watcher.Event, testFile, FS_MODIFY)

	if err = os.Rename(testFile, testFile2); err != nil {
		t.Fatalf("failed to rename test file: %s", err)
	}
	expect(t, watcher.Event, testFile, FS_MOVED_FROM)
	expect(t, watcher.Event, testFile2, FS_MOVED_TO)
	expect(t, watcher.Event, testFile, FS_MOVE_SELF)

	if err = os.RemoveAll(testDir); err != nil {
		t.Fatalf("failed to remove test directory: %s", err)
	}
	expect(t, watcher.Event, testFile2, FS_DELETE_SELF)
	expect(t, watcher.Event, testFile2, FS_IGNORED)
	expect(t, watcher.Event, testFile2, FS_DELETE)
	expect(t, watcher.Event, testDir, FS_DELETE_SELF)
	expect(t, watcher.Event, testDir, FS_IGNORED)

	t.Log("calling Close()")
	if err = watcher.Close(); err != nil {
		t.Fatalf("failed to close watcher: %s", err)
	}
}

func TestNotifyClose(t *testing.T) {
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
