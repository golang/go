// Copyright 2009 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build unix

package signal

import (
	"bytes"
	"context"
	"flag"
	"fmt"
	"internal/testenv"
	"os"
	"os/exec"
	"runtime"
	"runtime/trace"
	"strconv"
	"strings"
	"sync"
	"syscall"
	"testing"
	"time"
)

// settleTime is an upper bound on how long we expect signals to take to be
// delivered. Lower values make the test faster, but also flakier — especially
// on heavily loaded systems.
//
// The current value is set based on flakes observed in the Go builders.
var settleTime = 100 * time.Millisecond

// fatalWaitingTime is an absurdly long time to wait for signals to be
// delivered but, using it, we (hopefully) eliminate test flakes on the
// build servers. See #46736 for discussion.
var fatalWaitingTime = 30 * time.Second

func init() {
	if testenv.Builder() == "solaris-amd64-oraclerel" {
		// The solaris-amd64-oraclerel builder has been observed to time out in
		// TestNohup even with a 250ms settle time.
		//
		// Use a much longer settle time on that builder to try to suss out whether
		// the test is flaky due to builder slowness (which may mean we need a
		// longer GO_TEST_TIMEOUT_SCALE) or due to a dropped signal (which may
		// instead need a test-skip and upstream bug filed against the Solaris
		// kernel).
		//
		// See https://golang.org/issue/33174.
		settleTime = 5 * time.Second
	} else if runtime.GOOS == "linux" && strings.HasPrefix(runtime.GOARCH, "ppc64") {
		// Older linux kernels seem to have some hiccups delivering the signal
		// in a timely manner on ppc64 and ppc64le. When running on a
		// ppc64le/ubuntu 16.04/linux 4.4 host the time can vary quite
		// substantially even on an idle system. 5 seconds is twice any value
		// observed when running 10000 tests on such a system.
		settleTime = 5 * time.Second
	} else if s := os.Getenv("GO_TEST_TIMEOUT_SCALE"); s != "" {
		if scale, err := strconv.Atoi(s); err == nil {
			settleTime *= time.Duration(scale)
		}
	}
}

func waitSig(t *testing.T, c <-chan os.Signal, sig os.Signal) {
	t.Helper()
	waitSig1(t, c, sig, false)
}
func waitSigAll(t *testing.T, c <-chan os.Signal, sig os.Signal) {
	t.Helper()
	waitSig1(t, c, sig, true)
}

func waitSig1(t *testing.T, c <-chan os.Signal, sig os.Signal, all bool) {
	t.Helper()

	// Sleep multiple times to give the kernel more tries to
	// deliver the signal.
	start := time.Now()
	timer := time.NewTimer(settleTime / 10)
	defer timer.Stop()
	// If the caller notified for all signals on c, filter out SIGURG,
	// which is used for runtime preemption and can come at unpredictable times.
	// General user code should filter out all unexpected signals instead of just
	// SIGURG, but since os/signal is tightly coupled to the runtime it seems
	// appropriate to be stricter here.
	for time.Since(start) < fatalWaitingTime {
		select {
		case s := <-c:
			if s == sig {
				return
			}
			if !all || s != syscall.SIGURG {
				t.Fatalf("signal was %v, want %v", s, sig)
			}
		case <-timer.C:
			timer.Reset(settleTime / 10)
		}
	}
	t.Fatalf("timeout after %v waiting for %v", fatalWaitingTime, sig)
}

// quiesce waits until we can be reasonably confident that all pending signals
// have been delivered by the OS.
func quiesce() {
	// The kernel will deliver a signal as a thread returns
	// from a syscall. If the only active thread is sleeping,
	// and the system is busy, the kernel may not get around
	// to waking up a thread to catch the signal.
	// We try splitting up the sleep to give the kernel
	// many chances to deliver the signal.
	start := time.Now()
	for time.Since(start) < settleTime {
		time.Sleep(settleTime / 10)
	}
}

// Test that basic signal handling works.
func TestSignal(t *testing.T) {
	// Ask for SIGHUP
	c := make(chan os.Signal, 1)
	Notify(c, syscall.SIGHUP)
	defer Stop(c)

	// Send this process a SIGHUP
	t.Logf("sighup...")
	syscall.Kill(syscall.Getpid(), syscall.SIGHUP)
	waitSig(t, c, syscall.SIGHUP)

	// Ask for everything we can get. The buffer size has to be
	// more than 1, since the runtime might send SIGURG signals.
	// Using 10 is arbitrary.
	c1 := make(chan os.Signal, 10)
	Notify(c1)
	// Stop relaying the SIGURG signals. See #49724
	Reset(syscall.SIGURG)
	defer Stop(c1)

	// Send this process a SIGWINCH
	t.Logf("sigwinch...")
	syscall.Kill(syscall.Getpid(), syscall.SIGWINCH)
	waitSigAll(t, c1, syscall.SIGWINCH)

	// Send two more SIGHUPs, to make sure that
	// they get delivered on c1 and that not reading
	// from c does not block everything.
	t.Logf("sighup...")
	syscall.Kill(syscall.Getpid(), syscall.SIGHUP)
	waitSigAll(t, c1, syscall.SIGHUP)
	t.Logf("sighup...")
	syscall.Kill(syscall.Getpid(), syscall.SIGHUP)
	waitSigAll(t, c1, syscall.SIGHUP)

	// The first SIGHUP should be waiting for us on c.
	waitSig(t, c, syscall.SIGHUP)
}

func TestStress(t *testing.T) {
	dur := 3 * time.Second
	if testing.Short() {
		dur = 100 * time.Millisecond
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(4))

	sig := make(chan os.Signal, 1)
	Notify(sig, syscall.SIGUSR1)

	go func() {
		stop := time.After(dur)
		for {
			select {
			case <-stop:
				// Allow enough time for all signals to be delivered before we stop
				// listening for them.
				quiesce()
				Stop(sig)
				// According to its documentation, “[w]hen Stop returns, it in
				// guaranteed that c will receive no more signals.” So we can safely
				// close sig here: if there is a send-after-close race here, that is a
				// bug in Stop and we would like to detect it.
				close(sig)
				return

			default:
				syscall.Kill(syscall.Getpid(), syscall.SIGUSR1)
				runtime.Gosched()
			}
		}
	}()

	for range sig {
		// Receive signals until the sender closes sig.
	}
}

func testCancel(t *testing.T, ignore bool) {
	// Ask to be notified on c1 when a SIGWINCH is received.
	c1 := make(chan os.Signal, 1)
	Notify(c1, syscall.SIGWINCH)
	defer Stop(c1)

	// Ask to be notified on c2 when a SIGHUP is received.
	c2 := make(chan os.Signal, 1)
	Notify(c2, syscall.SIGHUP)
	defer Stop(c2)

	// Send this process a SIGWINCH and wait for notification on c1.
	syscall.Kill(syscall.Getpid(), syscall.SIGWINCH)
	waitSig(t, c1, syscall.SIGWINCH)

	// Send this process a SIGHUP and wait for notification on c2.
	syscall.Kill(syscall.Getpid(), syscall.SIGHUP)
	waitSig(t, c2, syscall.SIGHUP)

	// Ignore, or reset the signal handlers for, SIGWINCH and SIGHUP.
	// Either way, this should undo both calls to Notify above.
	if ignore {
		Ignore(syscall.SIGWINCH, syscall.SIGHUP)
		defer Reset(syscall.SIGWINCH, syscall.SIGHUP)
	} else {
		Reset(syscall.SIGWINCH, syscall.SIGHUP)
	}

	// Send this process a SIGWINCH. It should be ignored.
	syscall.Kill(syscall.Getpid(), syscall.SIGWINCH)

	// If ignoring, Send this process a SIGHUP. It should be ignored.
	if ignore {
		syscall.Kill(syscall.Getpid(), syscall.SIGHUP)
	}

	quiesce()

	select {
	case s := <-c1:
		t.Errorf("unexpected signal %v", s)
	default:
		// nothing to read - good
	}

	select {
	case s := <-c2:
		t.Errorf("unexpected signal %v", s)
	default:
		// nothing to read - good
	}

	// One or both of the signals may have been blocked for this process
	// by the calling process.
	// Discard any queued signals now to avoid interfering with other tests.
	Notify(c1, syscall.SIGWINCH)
	Notify(c2, syscall.SIGHUP)
	quiesce()
}

// Test that Reset cancels registration for listed signals on all channels.
func TestReset(t *testing.T) {
	testCancel(t, false)
}

// Test that Ignore cancels registration for listed signals on all channels.
func TestIgnore(t *testing.T) {
	testCancel(t, true)
}

// Test that Ignored correctly detects changes to the ignored status of a signal.
func TestIgnored(t *testing.T) {
	// Ask to be notified on SIGWINCH.
	c := make(chan os.Signal, 1)
	Notify(c, syscall.SIGWINCH)

	// If we're being notified, then the signal should not be ignored.
	if Ignored(syscall.SIGWINCH) {
		t.Errorf("expected SIGWINCH to not be ignored.")
	}
	Stop(c)
	Ignore(syscall.SIGWINCH)

	// We're no longer paying attention to this signal.
	if !Ignored(syscall.SIGWINCH) {
		t.Errorf("expected SIGWINCH to be ignored when explicitly ignoring it.")
	}

	Reset()
}

var checkSighupIgnored = flag.Bool("check_sighup_ignored", false, "if true, TestDetectNohup will fail if SIGHUP is not ignored.")

// Test that Ignored(SIGHUP) correctly detects whether it is being run under nohup.
func TestDetectNohup(t *testing.T) {
	if *checkSighupIgnored {
		if !Ignored(syscall.SIGHUP) {
			t.Fatal("SIGHUP is not ignored.")
		} else {
			t.Log("SIGHUP is ignored.")
		}
	} else {
		defer Reset()
		// Ugly: ask for SIGHUP so that child will not have no-hup set
		// even if test is running under nohup environment.
		// We have no intention of reading from c.
		c := make(chan os.Signal, 1)
		Notify(c, syscall.SIGHUP)
		if out, err := testenv.Command(t, os.Args[0], "-test.run=^TestDetectNohup$", "-check_sighup_ignored").CombinedOutput(); err == nil {
			t.Errorf("ran test with -check_sighup_ignored and it succeeded: expected failure.\nOutput:\n%s", out)
		}
		Stop(c)

		// Again, this time with nohup, assuming we can find it.
		_, err := os.Stat("/usr/bin/nohup")
		if err != nil {
			t.Skip("cannot find nohup; skipping second half of test")
		}
		Ignore(syscall.SIGHUP)
		os.Remove("nohup.out")
		out, err := testenv.Command(t, "/usr/bin/nohup", os.Args[0], "-test.run=^TestDetectNohup$", "-check_sighup_ignored").CombinedOutput()

		data, _ := os.ReadFile("nohup.out")
		os.Remove("nohup.out")
		if err != nil {
			// nohup doesn't work on new LUCI darwin builders due to the
			// type of launchd service the test run under. See
			// https://go.dev/issue/63875.
			if runtime.GOOS == "darwin" && strings.Contains(string(out), "nohup: can't detach from console: Inappropriate ioctl for device") {
				t.Skip("Skipping nohup test due to darwin builder limitation. See https://go.dev/issue/63875.")
			}

			t.Errorf("ran test with -check_sighup_ignored under nohup and it failed: expected success.\nError: %v\nOutput:\n%s%s", err, out, data)
		}
	}
}

var (
	sendUncaughtSighup = flag.Int("send_uncaught_sighup", 0, "send uncaught SIGHUP during TestStop")
	dieFromSighup      = flag.Bool("die_from_sighup", false, "wait to die from uncaught SIGHUP")
)

// Test that Stop cancels the channel's registrations.
func TestStop(t *testing.T) {
	sigs := []syscall.Signal{
		syscall.SIGWINCH,
		syscall.SIGHUP,
		syscall.SIGUSR1,
	}

	for _, sig := range sigs {
		sig := sig
		t.Run(fmt.Sprint(sig), func(t *testing.T) {
			defer Reset(sig)

			// When calling Notify with a specific signal,
			// independent signals should not interfere with each other,
			// and we end up needing to wait for signals to quiesce a lot.
			// Test the three different signals concurrently.
			t.Parallel()

			// If the signal is not ignored, send the signal before registering a
			// channel to verify the behavior of the default Go handler.
			// If it's SIGWINCH or SIGUSR1 we should not see it.
			// If it's SIGHUP, maybe we'll die. Let the flag tell us what to do.
			mayHaveBlockedSignal := false
			if !Ignored(sig) && (sig != syscall.SIGHUP || *sendUncaughtSighup == 1) {
				syscall.Kill(syscall.Getpid(), sig)
				quiesce()

				// We don't know whether sig is blocked for this process; see
				// https://golang.org/issue/38165. Assume that it could be.
				mayHaveBlockedSignal = true
			}

			// Ask for signal
			c := make(chan os.Signal, 1)
			Notify(c, sig)

			// Send this process the signal again.
			syscall.Kill(syscall.Getpid(), sig)
			waitSig(t, c, sig)

			if mayHaveBlockedSignal {
				// We may have received a queued initial signal in addition to the one
				// that we sent after Notify. If so, waitSig may have observed that
				// initial signal instead of the second one, and we may need to wait for
				// the second signal to clear. Do that now.
				quiesce()
				select {
				case <-c:
				default:
				}
			}

			// Stop watching for the signal and send it again.
			// If it's SIGHUP, maybe we'll die. Let the flag tell us what to do.
			Stop(c)
			if sig != syscall.SIGHUP || *sendUncaughtSighup == 2 {
				syscall.Kill(syscall.Getpid(), sig)
				quiesce()

				select {
				case s := <-c:
					t.Errorf("unexpected signal %v", s)
				default:
					// nothing to read - good
				}

				// If we're going to receive a signal, it has almost certainly been
				// received by now. However, it may have been blocked for this process —
				// we don't know. Explicitly unblock it and wait for it to clear now.
				Notify(c, sig)
				quiesce()
				Stop(c)
			}
		})
	}
}

// Test that when run under nohup, an uncaught SIGHUP does not kill the program.
func TestNohup(t *testing.T) {
	// When run without nohup, the test should crash on an uncaught SIGHUP.
	// When run under nohup, the test should ignore uncaught SIGHUPs,
	// because the runtime is not supposed to be listening for them.
	// Either way, TestStop should still be able to catch them when it wants them
	// and then when it stops wanting them, the original behavior should resume.
	//
	// send_uncaught_sighup=1 sends the SIGHUP before starting to listen for SIGHUPs.
	// send_uncaught_sighup=2 sends the SIGHUP after no longer listening for SIGHUPs.
	//
	// Both should fail without nohup and succeed with nohup.

	t.Run("uncaught", func(t *testing.T) {
		// Ugly: ask for SIGHUP so that child will not have no-hup set
		// even if test is running under nohup environment.
		// We have no intention of reading from c.
		c := make(chan os.Signal, 1)
		Notify(c, syscall.SIGHUP)
		t.Cleanup(func() { Stop(c) })

		var subTimeout time.Duration
		if deadline, ok := t.Deadline(); ok {
			subTimeout = time.Until(deadline)
			subTimeout -= subTimeout / 10 // Leave 10% headroom for propagating output.
		}
		for i := 1; i <= 2; i++ {
			i := i
			t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
				t.Parallel()

				args := []string{
					"-test.v",
					"-test.run=^TestStop$",
					"-send_uncaught_sighup=" + strconv.Itoa(i),
					"-die_from_sighup",
				}
				if subTimeout != 0 {
					args = append(args, fmt.Sprintf("-test.timeout=%v", subTimeout))
				}
				out, err := testenv.Command(t, os.Args[0], args...).CombinedOutput()

				if err == nil {
					t.Errorf("ran test with -send_uncaught_sighup=%d and it succeeded: expected failure.\nOutput:\n%s", i, out)
				} else {
					t.Logf("test with -send_uncaught_sighup=%d failed as expected.\nError: %v\nOutput:\n%s", i, err, out)
				}
			})
		}
	})

	t.Run("nohup", func(t *testing.T) {
		// Skip the nohup test below when running in tmux on darwin, since nohup
		// doesn't work correctly there. See issue #5135.
		if runtime.GOOS == "darwin" && os.Getenv("TMUX") != "" {
			t.Skip("Skipping nohup test due to running in tmux on darwin")
		}

		// Again, this time with nohup, assuming we can find it.
		_, err := exec.LookPath("nohup")
		if err != nil {
			t.Skip("cannot find nohup; skipping second half of test")
		}

		var subTimeout time.Duration
		if deadline, ok := t.Deadline(); ok {
			subTimeout = time.Until(deadline)
			subTimeout -= subTimeout / 10 // Leave 10% headroom for propagating output.
		}
		for i := 1; i <= 2; i++ {
			i := i
			t.Run(fmt.Sprintf("%d", i), func(t *testing.T) {
				t.Parallel()

				// POSIX specifies that nohup writes to a file named nohup.out if standard
				// output is a terminal. However, for an exec.Cmd, standard output is
				// not a terminal — so we don't need to read or remove that file (and,
				// indeed, cannot even create it if the current user is unable to write to
				// GOROOT/src, such as when GOROOT is installed and owned by root).

				args := []string{
					os.Args[0],
					"-test.v",
					"-test.run=^TestStop$",
					"-send_uncaught_sighup=" + strconv.Itoa(i),
				}
				if subTimeout != 0 {
					args = append(args, fmt.Sprintf("-test.timeout=%v", subTimeout))
				}
				out, err := testenv.Command(t, "nohup", args...).CombinedOutput()

				if err != nil {
					// nohup doesn't work on new LUCI darwin builders due to the
					// type of launchd service the test run under. See
					// https://go.dev/issue/63875.
					if runtime.GOOS == "darwin" && strings.Contains(string(out), "nohup: can't detach from console: Inappropriate ioctl for device") {
						// TODO(go.dev/issue/63799): A false-positive in vet reports a
						// t.Skip here as invalid. Switch back to t.Skip once fixed.
						t.Logf("Skipping nohup test due to darwin builder limitation. See https://go.dev/issue/63875.")
						return
					}

					t.Errorf("ran test with -send_uncaught_sighup=%d under nohup and it failed: expected success.\nError: %v\nOutput:\n%s", i, err, out)
				} else {
					t.Logf("ran test with -send_uncaught_sighup=%d under nohup.\nOutput:\n%s", i, out)
				}
			})
		}
	})
}

// Test that SIGCONT works (issue 8953).
func TestSIGCONT(t *testing.T) {
	c := make(chan os.Signal, 1)
	Notify(c, syscall.SIGCONT)
	defer Stop(c)
	syscall.Kill(syscall.Getpid(), syscall.SIGCONT)
	waitSig(t, c, syscall.SIGCONT)
}

// Test race between stopping and receiving a signal (issue 14571).
func TestAtomicStop(t *testing.T) {
	if os.Getenv("GO_TEST_ATOMIC_STOP") != "" {
		atomicStopTestProgram(t)
		t.Fatal("atomicStopTestProgram returned")
	}

	testenv.MustHaveExec(t)

	// Call Notify for SIGINT before starting the child process.
	// That ensures that SIGINT is not ignored for the child.
	// This is necessary because if SIGINT is ignored when a
	// Go program starts, then it remains ignored, and closing
	// the last notification channel for SIGINT will switch it
	// back to being ignored. In that case the assumption of
	// atomicStopTestProgram, that it will either die from SIGINT
	// or have it be reported, breaks down, as there is a third
	// option: SIGINT might be ignored.
	cs := make(chan os.Signal, 1)
	Notify(cs, syscall.SIGINT)
	defer Stop(cs)

	const execs = 10
	for i := 0; i < execs; i++ {
		timeout := "0"
		if deadline, ok := t.Deadline(); ok {
			timeout = time.Until(deadline).String()
		}
		cmd := testenv.Command(t, os.Args[0], "-test.run=^TestAtomicStop$", "-test.timeout="+timeout)
		cmd.Env = append(os.Environ(), "GO_TEST_ATOMIC_STOP=1")
		out, err := cmd.CombinedOutput()
		if err == nil {
			if len(out) > 0 {
				t.Logf("iteration %d: output %s", i, out)
			}
		} else {
			t.Logf("iteration %d: exit status %q: output: %s", i, err, out)
		}

		lost := bytes.Contains(out, []byte("lost signal"))
		if lost {
			t.Errorf("iteration %d: lost signal", i)
		}

		// The program should either die due to SIGINT,
		// or exit with success without printing "lost signal".
		if err == nil {
			if len(out) > 0 && !lost {
				t.Errorf("iteration %d: unexpected output", i)
			}
		} else {
			if ee, ok := err.(*exec.ExitError); !ok {
				t.Errorf("iteration %d: error (%v) has type %T; expected exec.ExitError", i, err, err)
			} else if ws, ok := ee.Sys().(syscall.WaitStatus); !ok {
				t.Errorf("iteration %d: error.Sys (%v) has type %T; expected syscall.WaitStatus", i, ee.Sys(), ee.Sys())
			} else if !ws.Signaled() || ws.Signal() != syscall.SIGINT {
				t.Errorf("iteration %d: got exit status %v; expected SIGINT", i, ee)
			}
		}
	}
}

// atomicStopTestProgram is run in a subprocess by TestAtomicStop.
// It tries to trigger a signal delivery race. This function should
// either catch a signal or die from it.
func atomicStopTestProgram(t *testing.T) {
	// This test won't work if SIGINT is ignored here.
	if Ignored(syscall.SIGINT) {
		fmt.Println("SIGINT is ignored")
		os.Exit(1)
	}

	const tries = 10

	timeout := 2 * time.Second
	if deadline, ok := t.Deadline(); ok {
		// Give each try an equal slice of the deadline, with one slice to spare for
		// cleanup.
		timeout = time.Until(deadline) / (tries + 1)
	}

	pid := syscall.Getpid()
	printed := false
	for i := 0; i < tries; i++ {
		cs := make(chan os.Signal, 1)
		Notify(cs, syscall.SIGINT)

		var wg sync.WaitGroup
		wg.Add(1)
		go func() {
			defer wg.Done()
			Stop(cs)
		}()

		syscall.Kill(pid, syscall.SIGINT)

		// At this point we should either die from SIGINT or
		// get a notification on cs. If neither happens, we
		// dropped the signal. It is given 2 seconds to
		// deliver, as needed for gccgo on some loaded test systems.

		select {
		case <-cs:
		case <-time.After(timeout):
			if !printed {
				fmt.Print("lost signal on tries:")
				printed = true
			}
			fmt.Printf(" %d", i)
		}

		wg.Wait()
	}
	if printed {
		fmt.Print("\n")
	}

	os.Exit(0)
}

func TestTime(t *testing.T) {
	// Test that signal works fine when we are in a call to get time,
	// which on some platforms is using VDSO. See issue #34391.
	dur := 3 * time.Second
	if testing.Short() {
		dur = 100 * time.Millisecond
	}
	defer runtime.GOMAXPROCS(runtime.GOMAXPROCS(4))

	sig := make(chan os.Signal, 1)
	Notify(sig, syscall.SIGUSR1)

	stop := make(chan struct{})
	go func() {
		for {
			select {
			case <-stop:
				// Allow enough time for all signals to be delivered before we stop
				// listening for them.
				quiesce()
				Stop(sig)
				// According to its documentation, “[w]hen Stop returns, it in
				// guaranteed that c will receive no more signals.” So we can safely
				// close sig here: if there is a send-after-close race, that is a bug in
				// Stop and we would like to detect it.
				close(sig)
				return

			default:
				syscall.Kill(syscall.Getpid(), syscall.SIGUSR1)
				runtime.Gosched()
			}
		}
	}()

	done := make(chan struct{})
	go func() {
		for range sig {
			// Receive signals until the sender closes sig.
		}
		close(done)
	}()

	t0 := time.Now()
	for t1 := t0; t1.Sub(t0) < dur; t1 = time.Now() {
	} // hammering on getting time

	close(stop)
	<-done
}

var (
	checkNotifyContext = flag.Bool("check_notify_ctx", false, "if true, TestNotifyContext will fail if SIGINT is not received.")
	ctxNotifyTimes     = flag.Int("ctx_notify_times", 1, "number of times a SIGINT signal should be received")
)

func TestNotifyContextNotifications(t *testing.T) {
	if *checkNotifyContext {
		ctx, _ := NotifyContext(context.Background(), syscall.SIGINT)
		// We want to make sure not to be calling Stop() internally on NotifyContext() when processing a received signal.
		// Being able to wait for a number of received system signals allows us to do so.
		var wg sync.WaitGroup
		n := *ctxNotifyTimes
		wg.Add(n)
		for i := 0; i < n; i++ {
			go func() {
				syscall.Kill(syscall.Getpid(), syscall.SIGINT)
				wg.Done()
			}()
		}
		wg.Wait()
		<-ctx.Done()
		fmt.Println("received SIGINT")
		// Sleep to give time to simultaneous signals to reach the process.
		// These signals must be ignored given stop() is not called on this code.
		// We want to guarantee a SIGINT doesn't cause a premature termination of the program.
		time.Sleep(settleTime)
		return
	}

	t.Parallel()
	testCases := []struct {
		name string
		n    int // number of times a SIGINT should be notified.
	}{
		{"once", 1},
		{"multiple", 10},
	}
	for _, tc := range testCases {
		tc := tc
		t.Run(tc.name, func(t *testing.T) {
			t.Parallel()

			var subTimeout time.Duration
			if deadline, ok := t.Deadline(); ok {
				timeout := time.Until(deadline)
				if timeout < 2*settleTime {
					t.Fatalf("starting test with less than %v remaining", 2*settleTime)
				}
				subTimeout = timeout - (timeout / 10) // Leave 10% headroom for cleaning up subprocess.
			}

			args := []string{
				"-test.v",
				"-test.run=^TestNotifyContextNotifications$",
				"-check_notify_ctx",
				fmt.Sprintf("-ctx_notify_times=%d", tc.n),
			}
			if subTimeout != 0 {
				args = append(args, fmt.Sprintf("-test.timeout=%v", subTimeout))
			}
			out, err := testenv.Command(t, os.Args[0], args...).CombinedOutput()
			if err != nil {
				t.Errorf("ran test with -check_notify_ctx_notification and it failed with %v.\nOutput:\n%s", err, out)
			}
			if want := []byte("received SIGINT\n"); !bytes.Contains(out, want) {
				t.Errorf("got %q, wanted %q", out, want)
			}
		})
	}
}

func TestNotifyContextStop(t *testing.T) {
	Ignore(syscall.SIGHUP)
	if !Ignored(syscall.SIGHUP) {
		t.Errorf("expected SIGHUP to be ignored when explicitly ignoring it.")
	}

	parent, cancelParent := context.WithCancel(context.Background())
	defer cancelParent()
	c, stop := NotifyContext(parent, syscall.SIGHUP)
	defer stop()

	// If we're being notified, then the signal should not be ignored.
	if Ignored(syscall.SIGHUP) {
		t.Errorf("expected SIGHUP to not be ignored.")
	}

	if want, got := "signal.NotifyContext(context.Background.WithCancel, [hangup])", fmt.Sprint(c); want != got {
		t.Errorf("c.String() = %q, wanted %q", got, want)
	}

	stop()
	<-c.Done()
	if got := c.Err(); got != context.Canceled {
		t.Errorf("c.Err() = %q, want %q", got, context.Canceled)
	}
}

func TestNotifyContextCancelParent(t *testing.T) {
	parent, cancelParent := context.WithCancel(context.Background())
	defer cancelParent()
	c, stop := NotifyContext(parent, syscall.SIGINT)
	defer stop()

	if want, got := "signal.NotifyContext(context.Background.WithCancel, [interrupt])", fmt.Sprint(c); want != got {
		t.Errorf("c.String() = %q, want %q", got, want)
	}

	cancelParent()
	<-c.Done()
	if got := c.Err(); got != context.Canceled {
		t.Errorf("c.Err() = %q, want %q", got, context.Canceled)
	}
}

func TestNotifyContextPrematureCancelParent(t *testing.T) {
	parent, cancelParent := context.WithCancel(context.Background())
	defer cancelParent()

	cancelParent() // Prematurely cancel context before calling NotifyContext.
	c, stop := NotifyContext(parent, syscall.SIGINT)
	defer stop()

	if want, got := "signal.NotifyContext(context.Background.WithCancel, [interrupt])", fmt.Sprint(c); want != got {
		t.Errorf("c.String() = %q, want %q", got, want)
	}

	<-c.Done()
	if got := c.Err(); got != context.Canceled {
		t.Errorf("c.Err() = %q, want %q", got, context.Canceled)
	}
}

func TestNotifyContextSimultaneousStop(t *testing.T) {
	c, stop := NotifyContext(context.Background(), syscall.SIGINT)
	defer stop()

	if want, got := "signal.NotifyContext(context.Background, [interrupt])", fmt.Sprint(c); want != got {
		t.Errorf("c.String() = %q, want %q", got, want)
	}

	var wg sync.WaitGroup
	n := 10
	wg.Add(n)
	for i := 0; i < n; i++ {
		go func() {
			stop()
			wg.Done()
		}()
	}
	wg.Wait()
	<-c.Done()
	if got := c.Err(); got != context.Canceled {
		t.Errorf("c.Err() = %q, want %q", got, context.Canceled)
	}
}

func TestNotifyContextStringer(t *testing.T) {
	parent, cancelParent := context.WithCancel(context.Background())
	defer cancelParent()
	c, stop := NotifyContext(parent, syscall.SIGHUP, syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	want := `signal.NotifyContext(context.Background.WithCancel, [hangup interrupt terminated])`
	if got := fmt.Sprint(c); got != want {
		t.Errorf("c.String() = %q, want %q", got, want)
	}
}

// #44193 test signal handling while stopping and starting the world.
func TestSignalTrace(t *testing.T) {
	done := make(chan struct{})
	quit := make(chan struct{})
	c := make(chan os.Signal, 1)
	Notify(c, syscall.SIGHUP)

	// Source and sink for signals busy loop unsynchronized with
	// trace starts and stops. We are ultimately validating that
	// signals and runtime.(stop|start)TheWorldGC are compatible.
	go func() {
		defer close(done)
		defer Stop(c)
		pid := syscall.Getpid()
		for {
			select {
			case <-quit:
				return
			default:
				syscall.Kill(pid, syscall.SIGHUP)
			}
			waitSig(t, c, syscall.SIGHUP)
		}
	}()

	for i := 0; i < 100; i++ {
		buf := new(bytes.Buffer)
		if err := trace.Start(buf); err != nil {
			t.Fatalf("[%d] failed to start tracing: %v", i, err)
		}
		trace.Stop()
		size := buf.Len()
		if size == 0 {
			t.Fatalf("[%d] trace is empty", i)
		}
	}
	close(quit)
	<-done
}

// #46321 test Reset actually undoes the effect of Ignore.
func TestResetIgnore(t *testing.T) {
	if os.Getenv("GO_TEST_RESET_IGNORE") != "" {
		s, err := strconv.Atoi(os.Getenv("GO_TEST_RESET_IGNORE"))
		if err != nil {
			t.Fatalf("failed to parse signal: %v", err)
		}
		if Ignored(syscall.Signal(s)) {
			os.Exit(1)
		}
		os.Exit(0)
	}

	sigs := []syscall.Signal{
		syscall.SIGHUP,
		syscall.SIGINT,
		syscall.SIGUSR1,
		syscall.SIGTERM,
		syscall.SIGCHLD,
		syscall.SIGWINCH,
	}

	for _, notify := range []bool{false, true} {
		for _, sig := range sigs {
			t.Run(fmt.Sprintf("%s[notify=%t]", sig, notify), func(t *testing.T) {
				if Ignored(sig) {
					t.Skipf("expected %q to not be ignored initially", sig)
				}

				Ignore(sig)
				if notify {
					c := make(chan os.Signal, 1)
					Notify(c, sig)
					defer Stop(c)
				}
				Reset(sig)

				if Ignored(sig) {
					t.Fatalf("expected %q to not be ignored", sig)
				}

				// Child processes inherit the ignored status of signals, so verify that it
				// is indeed not ignored.
				cmd := testenv.Command(t, testenv.Executable(t), "-test.run=^TestResetIgnore$")
				cmd.Env = append(os.Environ(), "GO_TEST_RESET_IGNORE="+strconv.Itoa(int(sig)))
				err := cmd.Run()
				if err != nil {
					t.Fatalf("expected %q to not be ignored in child process: %v", sig, err)
				}
			})
		}
	}
}

// #46321 test Reset correctly undoes the effect of Ignore when the child
// process is started with a signal ignored.
func TestInitiallyIgnoredResetIgnore(t *testing.T) {
	testenv.MustHaveExec(t)

	if os.Getenv("GO_TEST_INITIALLY_IGNORED_RESET_IGNORE") != "" {
		s, err := strconv.Atoi(os.Getenv("GO_TEST_INITIALLY_IGNORED_RESET_IGNORE"))
		if err != nil {
			t.Fatalf("failed to parse signal: %v", err)
		}
		initiallyIgnoredResetIgnoreTestProgram(syscall.Signal(s))
	}

	sigs := []syscall.Signal{
		syscall.SIGINT,
		syscall.SIGHUP,
	}

	for _, sig := range sigs {
		t.Run(fmt.Sprint(sig), func(t *testing.T) {
			Ignore(sig)
			defer Reset(sig)

			cmd := testenv.Command(t, testenv.Executable(t), "-test.run=^TestInitiallyIgnoredResetIgnore$")
			cmd.Env = append(os.Environ(), "GO_TEST_INITIALLY_IGNORED_RESET_IGNORE="+strconv.Itoa(int(sig)))
			err := cmd.Run()
			if err != nil {
				t.Fatalf("expected %q to be ignored in child process: %v", sig, err)
			}
		})
	}
}

func initiallyIgnoredResetIgnoreTestProgram(sig os.Signal) {
	if !Ignored(sig) {
		os.Exit(1)
	}
	Reset(sig)
	if !Ignored(sig) {
		os.Exit(1)
	}
	Ignore(sig)
	Reset(sig)
	if !Ignored(sig) {
		os.Exit(1)
	}
	os.Exit(0)
}
