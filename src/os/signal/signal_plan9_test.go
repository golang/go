// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package signal

import (
	"internal/itoa"
	"os"
	"runtime"
	"syscall"
	"testing"
	"time"
)

func waitSig(t *testing.T, c <-chan os.Signal, sig os.Signal) {
	select {
	case s := <-c:
		if s != sig {
			t.Fatalf("signal was %v, want %v", s, sig)
		}
	case <-time.After(1 * time.Second):
		t.Fatalf("timeout waiting for %v", sig)
	}
}

// Test that basic signal handling works.
func TestSignal(t *testing.T) {
	// Ask for hangup
	c := make(chan os.Signal, 1)
	Notify(c, syscall.Note("hangup"))
	defer Stop(c)

	// Send this process a hangup
	t.Logf("hangup...")
	postNote(syscall.Getpid(), "hangup")
	waitSig(t, c, syscall.Note("hangup"))

	// Ask for everything we can get.
	c1 := make(chan os.Signal, 1)
	Notify(c1)

	// Send this process an alarm
	t.Logf("alarm...")
	postNote(syscall.Getpid(), "alarm")
	waitSig(t, c1, syscall.Note("alarm"))

	// Send two more hangups, to make sure that
	// they get delivered on c1 and that not reading
	// from c does not block everything.
	t.Logf("hangup...")
	postNote(syscall.Getpid(), "hangup")
	waitSig(t, c1, syscall.Note("hangup"))
	t.Logf("hangup...")
	postNote(syscall.Getpid(), "hangup")
	waitSig(t, c1, syscall.Note("hangup"))

	// The first SIGHUP should be waiting for us on c.
	waitSig(t, c, syscall.Note("hangup"))
}

func TestStress(t *testing.T) {
	dur := 3 * time.Second
	if testing.Short() {
		dur = 100 * time.Millisecond
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(4))
	done := make(chan bool)
	finished := make(chan bool)
	go func() {
		sig := make(chan os.Signal, 1)
		Notify(sig, syscall.Note("alarm"))
		defer Stop(sig)
	Loop:
		for {
			select {
			case <-sig:
			case <-done:
				break Loop
			}
		}
		finished <- true
	}()
	go func() {
	Loop:
		for {
			select {
			case <-done:
				break Loop
			default:
				postNote(syscall.Getpid(), "alarm")
				runtime.Gosched()
			}
		}
		finished <- true
	}()
	time.Sleep(dur)
	close(done)
	<-finished
	<-finished
	// When run with 'go test -cpu=1,2,4' alarm from this test can slip
	// into subsequent TestSignal() causing failure.
	// Sleep for a while to reduce the possibility of the failure.
	time.Sleep(10 * time.Millisecond)
}

// Test that Stop cancels the channel's registrations.
func TestStop(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping in short mode")
	}
	sigs := []string{
		"alarm",
		"hangup",
	}

	for _, sig := range sigs {
		// Send the signal.
		// If it's alarm, we should not see it.
		// If it's hangup, maybe we'll die. Let the flag tell us what to do.
		if sig != "hangup" {
			postNote(syscall.Getpid(), sig)
		}
		time.Sleep(100 * time.Millisecond)

		// Ask for signal
		c := make(chan os.Signal, 1)
		Notify(c, syscall.Note(sig))
		defer Stop(c)

		// Send this process that signal
		postNote(syscall.Getpid(), sig)
		waitSig(t, c, syscall.Note(sig))

		Stop(c)
		select {
		case s := <-c:
			t.Fatalf("unexpected signal %v", s)
		case <-time.After(100 * time.Millisecond):
			// nothing to read - good
		}

		// Send the signal.
		// If it's alarm, we should not see it.
		// If it's hangup, maybe we'll die. Let the flag tell us what to do.
		if sig != "hangup" {
			postNote(syscall.Getpid(), sig)
		}

		select {
		case s := <-c:
			t.Fatalf("unexpected signal %v", s)
		case <-time.After(100 * time.Millisecond):
			// nothing to read - good
		}
	}
}

func postNote(pid int, note string) error {
	f, err := os.OpenFile("/proc/"+itoa.Itoa(pid)+"/note", os.O_WRONLY, 0)
	if err != nil {
		return err
	}
	defer f.Close()
	_, err = f.Write([]byte(note))
	return err
}
