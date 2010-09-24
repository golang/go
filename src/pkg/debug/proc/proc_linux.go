// Copyright 2009 The Go Authors.  All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package proc

// TODO(rsc): Imports here after to be in proc.go too in order
// for deps.bash to get the right answer.
import (
	"container/vector"
	"fmt"
	"io/ioutil"
	"os"
	"runtime"
	"strconv"
	"strings"
	"sync"
	"syscall"
)

// This is an implementation of the process tracing interface using
// Linux's ptrace(2) interface.  The implementation is multi-threaded.
// Each attached process has an associated monitor thread, and each
// running attached thread has an associated "wait" thread.  The wait
// thread calls wait4 on the thread's TID and reports any wait events
// or errors via "debug events".  The monitor thread consumes these
// wait events and updates the internally maintained state of each
// thread.  All ptrace calls must run in the monitor thread, so the
// monitor executes closures received on the debugReq channel.
//
// As ptrace's documentation is somewhat light, this is heavily based
// on information gleaned from the implementation of ptrace found at
//   http://lxr.linux.no/linux+v2.6.30/kernel/ptrace.c
//   http://lxr.linux.no/linux+v2.6.30/arch/x86/kernel/ptrace.c#L854
// as well as experimentation and examination of gdb's behavior.

const (
	trace    = false
	traceIP  = false
	traceMem = false
)

/*
 * Thread state
 */

// Each thread can be in one of the following set of states.
// Each state satisfies
//  isRunning() || isStopped() || isZombie() || isTerminal().
//
// Running threads can be sent signals and must be waited on, but they
// cannot be inspected using ptrace.
//
// Stopped threads can be inspected and continued, but cannot be
// meaningfully waited on.  They can be sent signals, but the signals
// will be queued until they are running again.
//
// Zombie threads cannot be inspected, continued, or sent signals (and
// therefore they cannot be stopped), but they must be waited on.
//
// Terminal threads no longer exist in the OS and thus you can't do
// anything with them.
type threadState string

const (
	running             threadState = "Running"
	singleStepping      threadState = "SingleStepping" // Transient
	stopping            threadState = "Stopping"       // Transient
	stopped             threadState = "Stopped"
	stoppedBreakpoint   threadState = "StoppedBreakpoint"
	stoppedSignal       threadState = "StoppedSignal"
	stoppedThreadCreate threadState = "StoppedThreadCreate"
	stoppedExiting      threadState = "StoppedExiting"
	exiting             threadState = "Exiting" // Transient (except main thread)
	exited              threadState = "Exited"
	detached            threadState = "Detached"
)

func (ts threadState) isRunning() bool {
	return ts == running || ts == singleStepping || ts == stopping
}

func (ts threadState) isStopped() bool {
	return ts == stopped || ts == stoppedBreakpoint || ts == stoppedSignal || ts == stoppedThreadCreate || ts == stoppedExiting
}

func (ts threadState) isZombie() bool { return ts == exiting }

func (ts threadState) isTerminal() bool { return ts == exited || ts == detached }

func (ts threadState) String() string { return string(ts) }

/*
 * Basic types
 */

// A breakpoint stores information about a single breakpoint,
// including its program counter, the overwritten text if the
// breakpoint is installed.
type breakpoint struct {
	pc      uintptr
	olddata []byte
}

func (bp *breakpoint) String() string {
	if bp == nil {
		return "<nil>"
	}
	return fmt.Sprintf("%#x", bp.pc)
}

// bpinst386 is the breakpoint instruction used on 386 and amd64.
var bpinst386 = []byte{0xcc}

// A debugEvent represents a reason a thread stopped or a wait error.
type debugEvent struct {
	*os.Waitmsg
	t   *thread
	err os.Error
}

// A debugReq is a request to execute a closure in the monitor thread.
type debugReq struct {
	f   func() os.Error
	res chan os.Error
}

// A transitionHandler specifies a function to be called when a thread
// changes state and a function to be called when an error occurs in
// the monitor.  Both run in the monitor thread.  Before the monitor
// invokes a handler, it removes the handler from the handler queue.
// The handler should re-add itself if needed.
type transitionHandler struct {
	handle func(*thread, threadState, threadState)
	onErr  func(os.Error)
}

// A process is a Linux process, which consists of a set of threads.
// Each running process has one monitor thread, which processes
// messages from the debugEvents, debugReqs, and stopReq channels and
// calls transition handlers.
//
// To send a message to the monitor thread, first receive from the
// ready channel.  If the ready channel returns true, the monitor is
// still running and will accept a message.  If the ready channel
// returns false, the monitor is not running (the ready channel has
// been closed), and the reason it is not running will be stored in err.
type process struct {
	pid                int
	threads            map[int]*thread
	breakpoints        map[uintptr]*breakpoint
	ready              chan bool
	debugEvents        chan *debugEvent
	debugReqs          chan *debugReq
	stopReq            chan os.Error
	transitionHandlers vector.Vector
	err                os.Error
}

// A thread represents a Linux thread in another process that is being
// debugged.  Each running thread has an associated goroutine that
// waits for thread updates and sends them to the process monitor.
type thread struct {
	tid  int
	proc *process
	// Whether to ignore the next SIGSTOP received by wait.
	ignoreNextSigstop bool

	// Thread state.  Only modified via setState.
	state threadState
	// If state == StoppedBreakpoint
	breakpoint *breakpoint
	// If state == StoppedSignal or state == Exited
	signal int
	// If state == StoppedThreadCreate
	newThread *thread
	// If state == Exited
	exitStatus int
}

/*
 * Errors
 */

type badState struct {
	thread  *thread
	message string
	state   threadState
}

func (e *badState) String() string {
	return fmt.Sprintf("Thread %d %s from state %v", e.thread.tid, e.message, e.state)
}

type breakpointExistsError Word

func (e breakpointExistsError) String() string {
	return fmt.Sprintf("breakpoint already exists at PC %#x", e)
}

type noBreakpointError Word

func (e noBreakpointError) String() string { return fmt.Sprintf("no breakpoint at PC %#x", e) }

type newThreadError struct {
	*os.Waitmsg
	wantPid int
	wantSig int
}

func (e *newThreadError) String() string {
	return fmt.Sprintf("newThread wait wanted pid %v and signal %v, got %v and %v", e.Pid, e.StopSignal(), e.wantPid, e.wantSig)
}

type ProcessExited struct{}

func (p ProcessExited) String() string { return "process exited" }

/*
 * Ptrace wrappers
 */

func (t *thread) ptracePeekText(addr uintptr, out []byte) (int, os.Error) {
	c, err := syscall.PtracePeekText(t.tid, addr, out)
	if traceMem {
		fmt.Printf("peek(%#x) => %v, %v\n", addr, out, err)
	}
	return c, os.NewSyscallError("ptrace(PEEKTEXT)", err)
}

func (t *thread) ptracePokeText(addr uintptr, out []byte) (int, os.Error) {
	c, err := syscall.PtracePokeText(t.tid, addr, out)
	if traceMem {
		fmt.Printf("poke(%#x, %v) => %v\n", addr, out, err)
	}
	return c, os.NewSyscallError("ptrace(POKETEXT)", err)
}

func (t *thread) ptraceGetRegs(regs *syscall.PtraceRegs) os.Error {
	err := syscall.PtraceGetRegs(t.tid, regs)
	return os.NewSyscallError("ptrace(GETREGS)", err)
}

func (t *thread) ptraceSetRegs(regs *syscall.PtraceRegs) os.Error {
	err := syscall.PtraceSetRegs(t.tid, regs)
	return os.NewSyscallError("ptrace(SETREGS)", err)
}

func (t *thread) ptraceSetOptions(options int) os.Error {
	err := syscall.PtraceSetOptions(t.tid, options)
	return os.NewSyscallError("ptrace(SETOPTIONS)", err)
}

func (t *thread) ptraceGetEventMsg() (uint, os.Error) {
	msg, err := syscall.PtraceGetEventMsg(t.tid)
	return msg, os.NewSyscallError("ptrace(GETEVENTMSG)", err)
}

func (t *thread) ptraceCont() os.Error {
	err := syscall.PtraceCont(t.tid, 0)
	return os.NewSyscallError("ptrace(CONT)", err)
}

func (t *thread) ptraceContWithSignal(sig int) os.Error {
	err := syscall.PtraceCont(t.tid, sig)
	return os.NewSyscallError("ptrace(CONT)", err)
}

func (t *thread) ptraceStep() os.Error {
	err := syscall.PtraceSingleStep(t.tid)
	return os.NewSyscallError("ptrace(SINGLESTEP)", err)
}

func (t *thread) ptraceDetach() os.Error {
	err := syscall.PtraceDetach(t.tid)
	return os.NewSyscallError("ptrace(DETACH)", err)
}

/*
 * Logging utilties
 */

var logLock sync.Mutex

func (t *thread) logTrace(format string, args ...interface{}) {
	if !trace {
		return
	}
	logLock.Lock()
	defer logLock.Unlock()
	fmt.Fprintf(os.Stderr, "Thread %d", t.tid)
	if traceIP {
		var regs syscall.PtraceRegs
		err := t.ptraceGetRegs(&regs)
		if err == nil {
			fmt.Fprintf(os.Stderr, "@%x", regs.PC())
		}
	}
	fmt.Fprint(os.Stderr, ": ")
	fmt.Fprintf(os.Stderr, format, args...)
	fmt.Fprint(os.Stderr, "\n")
}

func (t *thread) warn(format string, args ...interface{}) {
	logLock.Lock()
	defer logLock.Unlock()
	fmt.Fprintf(os.Stderr, "Thread %d: WARNING ", t.tid)
	fmt.Fprintf(os.Stderr, format, args...)
	fmt.Fprint(os.Stderr, "\n")
}

func (p *process) logTrace(format string, args ...interface{}) {
	if !trace {
		return
	}
	logLock.Lock()
	defer logLock.Unlock()
	fmt.Fprintf(os.Stderr, "Process %d: ", p.pid)
	fmt.Fprintf(os.Stderr, format, args...)
	fmt.Fprint(os.Stderr, "\n")
}

/*
 * State utilities
 */

// someStoppedThread returns a stopped thread from the process.
// Returns nil if no threads are stopped.
//
// Must be called from the monitor thread.
func (p *process) someStoppedThread() *thread {
	for _, t := range p.threads {
		if t.state.isStopped() {
			return t
		}
	}
	return nil
}

// someRunningThread returns a running thread from the process.
// Returns nil if no threads are running.
//
// Must be called from the monitor thread.
func (p *process) someRunningThread() *thread {
	for _, t := range p.threads {
		if t.state.isRunning() {
			return t
		}
	}
	return nil
}

/*
 * Breakpoint utilities
 */

// installBreakpoints adds breakpoints to the attached process.
//
// Must be called from the monitor thread.
func (p *process) installBreakpoints() os.Error {
	n := 0
	main := p.someStoppedThread()
	for _, b := range p.breakpoints {
		if b.olddata != nil {
			continue
		}

		b.olddata = make([]byte, len(bpinst386))
		_, err := main.ptracePeekText(uintptr(b.pc), b.olddata)
		if err != nil {
			b.olddata = nil
			return err
		}

		_, err = main.ptracePokeText(uintptr(b.pc), bpinst386)
		if err != nil {
			b.olddata = nil
			return err
		}
		n++
	}
	if n > 0 {
		p.logTrace("installed %d/%d breakpoints", n, len(p.breakpoints))
	}

	return nil
}

// uninstallBreakpoints removes the installed breakpoints from p.
//
// Must be called from the monitor thread.
func (p *process) uninstallBreakpoints() os.Error {
	if len(p.threads) == 0 {
		return nil
	}
	n := 0
	main := p.someStoppedThread()
	for _, b := range p.breakpoints {
		if b.olddata == nil {
			continue
		}

		_, err := main.ptracePokeText(uintptr(b.pc), b.olddata)
		if err != nil {
			return err
		}
		b.olddata = nil
		n++
	}
	if n > 0 {
		p.logTrace("uninstalled %d/%d breakpoints", n, len(p.breakpoints))
	}

	return nil
}

/*
 * Debug event handling
 */

// wait waits for a wait event from this thread and sends it on the
// debug events channel for this thread's process.  This should be
// started in its own goroutine when the attached thread enters a
// running state.  The goroutine will exit as soon as it sends a debug
// event.
func (t *thread) wait() {
	for {
		var ev debugEvent
		ev.t = t
		t.logTrace("beginning wait")
		ev.Waitmsg, ev.err = os.Wait(t.tid, syscall.WALL)
		if ev.err == nil && ev.Pid != t.tid {
			panic(fmt.Sprint("Wait returned pid ", ev.Pid, " wanted ", t.tid))
		}
		if ev.StopSignal() == syscall.SIGSTOP && t.ignoreNextSigstop {
			// Spurious SIGSTOP.  See Thread.Stop().
			t.ignoreNextSigstop = false
			err := t.ptraceCont()
			if err == nil {
				continue
			}
			// If we failed to continue, just let
			// the stop go through so we can
			// update the thread's state.
		}
		if !<-t.proc.ready {
			// The monitor exited
			break
		}
		t.proc.debugEvents <- &ev
		break
	}
}

// setState sets this thread's state, starts a wait thread if
// necessary, and invokes state transition handlers.
//
// Must be called from the monitor thread.
func (t *thread) setState(newState threadState) {
	oldState := t.state
	t.state = newState
	t.logTrace("state %v -> %v", oldState, newState)

	if !oldState.isRunning() && (newState.isRunning() || newState.isZombie()) {
		// Start waiting on this thread
		go t.wait()
	}

	// Invoke state change handlers
	handlers := t.proc.transitionHandlers
	if handlers.Len() == 0 {
		return
	}

	t.proc.transitionHandlers = nil
	for _, h := range handlers {
		h := h.(*transitionHandler)
		h.handle(t, oldState, newState)
	}
}

// sendSigstop sends a SIGSTOP to this thread.
func (t *thread) sendSigstop() os.Error {
	t.logTrace("sending SIGSTOP")
	err := syscall.Tgkill(t.proc.pid, t.tid, syscall.SIGSTOP)
	return os.NewSyscallError("tgkill", err)
}

// stopAsync sends SIGSTOP to all threads in state 'running'.
//
// Must be called from the monitor thread.
func (p *process) stopAsync() os.Error {
	for _, t := range p.threads {
		if t.state == running {
			err := t.sendSigstop()
			if err != nil {
				return err
			}
			t.setState(stopping)
		}
	}
	return nil
}

// doTrap handles SIGTRAP debug events with a cause of 0.  These can
// be caused either by an installed breakpoint, a breakpoint in the
// program text, or by single stepping.
//
// TODO(austin) I think we also get this on an execve syscall.
func (ev *debugEvent) doTrap() (threadState, os.Error) {
	t := ev.t

	if t.state == singleStepping {
		return stopped, nil
	}

	// Hit a breakpoint.  Linux leaves the program counter after
	// the breakpoint.  If this is an installed breakpoint, we
	// need to back the PC up to the breakpoint PC.
	var regs syscall.PtraceRegs
	err := t.ptraceGetRegs(&regs)
	if err != nil {
		return stopped, err
	}

	b, ok := t.proc.breakpoints[uintptr(regs.PC())-uintptr(len(bpinst386))]
	if !ok {
		// We must have hit a breakpoint that was actually in
		// the program.  Leave the IP where it is so we don't
		// re-execute the breakpoint instruction.  Expose the
		// fact that we stopped with a SIGTRAP.
		return stoppedSignal, nil
	}

	t.breakpoint = b
	t.logTrace("at breakpoint %v, backing up PC from %#x", b, regs.PC())

	regs.SetPC(uint64(b.pc))
	err = t.ptraceSetRegs(&regs)
	if err != nil {
		return stopped, err
	}
	return stoppedBreakpoint, nil
}

// doPtraceClone handles SIGTRAP debug events with a PTRACE_EVENT_CLONE
// cause.  It initializes the new thread, adds it to the process, and
// returns the appropriate thread state for the existing thread.
func (ev *debugEvent) doPtraceClone() (threadState, os.Error) {
	t := ev.t

	// Get the TID of the new thread
	tid, err := t.ptraceGetEventMsg()
	if err != nil {
		return stopped, err
	}

	nt, err := t.proc.newThread(int(tid), syscall.SIGSTOP, true)
	if err != nil {
		return stopped, err
	}

	// Remember the thread
	t.newThread = nt

	return stoppedThreadCreate, nil
}

// doPtraceExit handles SIGTRAP debug events with a PTRACE_EVENT_EXIT
// cause.  It sets up the thread's state, but does not remove it from
// the process.  A later WIFEXITED debug event will remove it from the
// process.
func (ev *debugEvent) doPtraceExit() (threadState, os.Error) {
	t := ev.t

	// Get exit status
	exitStatus, err := t.ptraceGetEventMsg()
	if err != nil {
		return stopped, err
	}
	ws := syscall.WaitStatus(exitStatus)
	t.logTrace("exited with %v", ws)
	switch {
	case ws.Exited():
		t.exitStatus = ws.ExitStatus()
	case ws.Signaled():
		t.signal = ws.Signal()
	}

	// We still need to continue this thread and wait on this
	// thread's WIFEXITED event.  We'll delete it then.
	return stoppedExiting, nil
}

// process handles a debug event.  It modifies any thread or process
// state as necessary, uninstalls breakpoints if necessary, and stops
// any running threads.
func (ev *debugEvent) process() os.Error {
	if ev.err != nil {
		return ev.err
	}

	t := ev.t
	t.exitStatus = -1
	t.signal = -1

	// Decode wait status.
	var state threadState
	switch {
	case ev.Stopped():
		state = stoppedSignal
		t.signal = ev.StopSignal()
		t.logTrace("stopped with %v", ev)
		if ev.StopSignal() == syscall.SIGTRAP {
			// What caused the debug trap?
			var err os.Error
			switch cause := ev.TrapCause(); cause {
			case 0:
				// Breakpoint or single stepping
				state, err = ev.doTrap()

			case syscall.PTRACE_EVENT_CLONE:
				state, err = ev.doPtraceClone()

			case syscall.PTRACE_EVENT_EXIT:
				state, err = ev.doPtraceExit()

			default:
				t.warn("Unknown trap cause %d", cause)
			}

			if err != nil {
				t.setState(stopped)
				t.warn("failed to handle trap %v: %v", ev, err)
			}
		}

	case ev.Exited():
		state = exited
		t.proc.threads[t.tid] = nil, false
		t.logTrace("exited %v", ev)
		// We should have gotten the exit status in
		// PTRACE_EVENT_EXIT, but just in case.
		t.exitStatus = ev.ExitStatus()

	case ev.Signaled():
		state = exited
		t.proc.threads[t.tid] = nil, false
		t.logTrace("signaled %v", ev)
		// Again, this should be redundant.
		t.signal = ev.Signal()

	default:
		panic(fmt.Sprintf("Unexpected wait status %v", ev.Waitmsg))
	}

	// If we sent a SIGSTOP to the thread (indicated by state
	// Stopping), we might have raced with a different type of
	// stop.  If we didn't get the stop we expected, then the
	// SIGSTOP we sent is now queued up, so we should ignore the
	// next one we get.
	if t.state == stopping && ev.StopSignal() != syscall.SIGSTOP {
		t.ignoreNextSigstop = true
	}

	// TODO(austin) If we're in state stopping and get a SIGSTOP,
	// set state stopped instead of stoppedSignal.

	t.setState(state)

	if t.proc.someRunningThread() == nil {
		// Nothing is running, uninstall breakpoints
		return t.proc.uninstallBreakpoints()
	}
	// Stop any other running threads
	return t.proc.stopAsync()
}

// onStop adds a handler for state transitions from running to
// non-running states.  The handler will be called from the monitor
// thread.
//
// Must be called from the monitor thread.
func (t *thread) onStop(handle func(), onErr func(os.Error)) {
	// TODO(austin) This is rather inefficient for things like
	// stepping all threads during a continue.  Maybe move
	// transitionHandlers to the thread, or have both per-thread
	// and per-process transition handlers.
	h := &transitionHandler{nil, onErr}
	h.handle = func(st *thread, old, new threadState) {
		if t == st && old.isRunning() && !new.isRunning() {
			handle()
		} else {
			t.proc.transitionHandlers.Push(h)
		}
	}
	t.proc.transitionHandlers.Push(h)
}

/*
 * Event monitor
 */

// monitor handles debug events and debug requests for p, exiting when
// there are no threads left in p.
func (p *process) monitor() {
	var err os.Error

	// Linux requires that all ptrace calls come from the thread
	// that originally attached.  Prevent the Go scheduler from
	// migrating us to other OS threads.
	runtime.LockOSThread()
	defer runtime.UnlockOSThread()

	hadThreads := false
	for err == nil {
		p.ready <- true
		select {
		case event := <-p.debugEvents:
			err = event.process()

		case req := <-p.debugReqs:
			req.res <- req.f()

		case err = <-p.stopReq:
			break
		}

		if len(p.threads) == 0 {
			if err == nil && hadThreads {
				p.logTrace("no more threads; monitor exiting")
				err = ProcessExited{}
			}
		} else {
			hadThreads = true
		}
	}

	// Abort waiting handlers
	// TODO(austin) How do I stop the wait threads?
	for _, h := range p.transitionHandlers {
		h := h.(*transitionHandler)
		h.onErr(err)
	}

	// Indicate that the monitor cannot receive any more messages
	p.err = err
	close(p.ready)
}

// do executes f in the monitor thread (and, thus, atomically with
// respect to thread state changes).  f must not block.
//
// Must NOT be called from the monitor thread.
func (p *process) do(f func() os.Error) os.Error {
	if !<-p.ready {
		return p.err
	}
	req := &debugReq{f, make(chan os.Error)}
	p.debugReqs <- req
	return <-req.res
}

// stopMonitor stops the monitor with the given error.  If the monitor
// is already stopped, does nothing.
func (p *process) stopMonitor(err os.Error) {
	if err == nil {
		panic("cannot stop the monitor with no error")
	}
	if <-p.ready {
		p.stopReq <- err
	}
}

/*
 * Public thread interface
 */

func (t *thread) Regs() (Regs, os.Error) {
	var regs syscall.PtraceRegs

	err := t.proc.do(func() os.Error {
		if !t.state.isStopped() {
			return &badState{t, "cannot get registers", t.state}
		}
		return t.ptraceGetRegs(&regs)
	})
	if err != nil {
		return nil, err
	}

	setter := func(r *syscall.PtraceRegs) os.Error {
		return t.proc.do(func() os.Error {
			if !t.state.isStopped() {
				return &badState{t, "cannot get registers", t.state}
			}
			return t.ptraceSetRegs(r)
		})
	}
	return newRegs(&regs, setter), nil
}

func (t *thread) Peek(addr Word, out []byte) (int, os.Error) {
	var c int

	err := t.proc.do(func() os.Error {
		if !t.state.isStopped() {
			return &badState{t, "cannot peek text", t.state}
		}

		var err os.Error
		c, err = t.ptracePeekText(uintptr(addr), out)
		return err
	})

	return c, err
}

func (t *thread) Poke(addr Word, out []byte) (int, os.Error) {
	var c int

	err := t.proc.do(func() os.Error {
		if !t.state.isStopped() {
			return &badState{t, "cannot poke text", t.state}
		}

		var err os.Error
		c, err = t.ptracePokeText(uintptr(addr), out)
		return err
	})

	return c, err
}

// stepAsync starts this thread single stepping.  When the single step
// is complete, it will send nil on the given channel.  If an error
// occurs while setting up the single step, it returns that error.  If
// an error occurs while waiting for the single step to complete, it
// sends that error on the channel.
func (t *thread) stepAsync(ready chan os.Error) os.Error {
	if err := t.ptraceStep(); err != nil {
		return err
	}
	t.setState(singleStepping)
	t.onStop(func() { ready <- nil },
		func(err os.Error) { ready <- err })
	return nil
}

func (t *thread) Step() os.Error {
	t.logTrace("Step {")
	defer t.logTrace("}")

	ready := make(chan os.Error)

	err := t.proc.do(func() os.Error {
		if !t.state.isStopped() {
			return &badState{t, "cannot single step", t.state}
		}
		return t.stepAsync(ready)
	})
	if err != nil {
		return err
	}

	err = <-ready
	return err
}

// TODO(austin) We should probably get this via C's strsignal.
var sigNames = [...]string{
	"SIGEXIT", "SIGHUP", "SIGINT", "SIGQUIT", "SIGILL",
	"SIGTRAP", "SIGABRT", "SIGBUS", "SIGFPE", "SIGKILL",
	"SIGUSR1", "SIGSEGV", "SIGUSR2", "SIGPIPE", "SIGALRM",
	"SIGTERM", "SIGSTKFLT", "SIGCHLD", "SIGCONT", "SIGSTOP",
	"SIGTSTP", "SIGTTIN", "SIGTTOU", "SIGURG", "SIGXCPU",
	"SIGXFSZ", "SIGVTALRM", "SIGPROF", "SIGWINCH", "SIGPOLL",
	"SIGPWR", "SIGSYS",
}

// sigName returns the symbolic name for the given signal number.  If
// the signal number is invalid, returns "<invalid>".
func sigName(signal int) string {
	if signal < 0 || signal >= len(sigNames) {
		return "<invalid>"
	}
	return sigNames[signal]
}

func (t *thread) Stopped() (Cause, os.Error) {
	var c Cause
	err := t.proc.do(func() os.Error {
		switch t.state {
		case stopped:
			c = Stopped{}

		case stoppedBreakpoint:
			c = Breakpoint(t.breakpoint.pc)

		case stoppedSignal:
			c = Signal(sigName(t.signal))

		case stoppedThreadCreate:
			c = &ThreadCreate{t.newThread}

		case stoppedExiting, exiting, exited:
			if t.signal == -1 {
				c = &ThreadExit{t.exitStatus, ""}
			} else {
				c = &ThreadExit{t.exitStatus, sigName(t.signal)}
			}

		default:
			return &badState{t, "cannot get stop cause", t.state}
		}
		return nil
	})
	if err != nil {
		return nil, err
	}

	return c, nil
}

func (p *process) Threads() []Thread {
	var res []Thread

	p.do(func() os.Error {
		res = make([]Thread, len(p.threads))
		i := 0
		for _, t := range p.threads {
			// Exclude zombie threads.
			st := t.state
			if st == exiting || st == exited || st == detached {
				continue
			}

			res[i] = t
			i++
		}
		res = res[0:i]
		return nil
	})
	return res
}

func (p *process) AddBreakpoint(pc Word) os.Error {
	return p.do(func() os.Error {
		if t := p.someRunningThread(); t != nil {
			return &badState{t, "cannot add breakpoint", t.state}
		}
		if _, ok := p.breakpoints[uintptr(pc)]; ok {
			return breakpointExistsError(pc)
		}
		p.breakpoints[uintptr(pc)] = &breakpoint{pc: uintptr(pc)}
		return nil
	})
}

func (p *process) RemoveBreakpoint(pc Word) os.Error {
	return p.do(func() os.Error {
		if t := p.someRunningThread(); t != nil {
			return &badState{t, "cannot remove breakpoint", t.state}
		}
		if _, ok := p.breakpoints[uintptr(pc)]; !ok {
			return noBreakpointError(pc)
		}
		p.breakpoints[uintptr(pc)] = nil, false
		return nil
	})
}

func (p *process) Continue() os.Error {
	// Single step any threads that are stopped at breakpoints so
	// we can reinstall breakpoints.
	var ready chan os.Error
	count := 0

	err := p.do(func() os.Error {
		// We make the ready channel big enough to hold all
		// ready message so we don't jam up the monitor if we
		// stop listening (e.g., if there's an error).
		ready = make(chan os.Error, len(p.threads))

		for _, t := range p.threads {
			if !t.state.isStopped() {
				continue
			}

			// We use the breakpoint map directly here
			// instead of checking the stop cause because
			// it could have been stopped at a breakpoint
			// for some other reason, or the breakpoint
			// could have been added since it was stopped.
			var regs syscall.PtraceRegs
			err := t.ptraceGetRegs(&regs)
			if err != nil {
				return err
			}
			if b, ok := p.breakpoints[uintptr(regs.PC())]; ok {
				t.logTrace("stepping over breakpoint %v", b)
				if err := t.stepAsync(ready); err != nil {
					return err
				}
				count++
			}
		}
		return nil
	})
	if err != nil {
		p.stopMonitor(err)
		return err
	}

	// Wait for single stepping threads
	for count > 0 {
		err = <-ready
		if err != nil {
			p.stopMonitor(err)
			return err
		}
		count--
	}

	// Continue all threads
	err = p.do(func() os.Error {
		if err := p.installBreakpoints(); err != nil {
			return err
		}

		for _, t := range p.threads {
			var err os.Error
			switch {
			case !t.state.isStopped():
				continue

			case t.state == stoppedSignal && t.signal != syscall.SIGSTOP && t.signal != syscall.SIGTRAP:
				t.logTrace("continuing with signal %d", t.signal)
				err = t.ptraceContWithSignal(t.signal)

			default:
				t.logTrace("continuing")
				err = t.ptraceCont()
			}
			if err != nil {
				return err
			}
			if t.state == stoppedExiting {
				t.setState(exiting)
			} else {
				t.setState(running)
			}
		}
		return nil
	})
	if err != nil {
		// TODO(austin) Do we need to stop the monitor with
		// this error atomically with the do-routine above?
		p.stopMonitor(err)
		return err
	}

	return nil
}

func (p *process) WaitStop() os.Error {
	// We need a non-blocking ready channel for the case where all
	// threads are already stopped.
	ready := make(chan os.Error, 1)

	err := p.do(func() os.Error {
		// Are all of the threads already stopped?
		if p.someRunningThread() == nil {
			ready <- nil
			return nil
		}

		// Monitor state transitions
		h := &transitionHandler{}
		h.handle = func(st *thread, old, new threadState) {
			if !new.isRunning() {
				if p.someRunningThread() == nil {
					ready <- nil
					return
				}
			}
			p.transitionHandlers.Push(h)
		}
		h.onErr = func(err os.Error) { ready <- err }
		p.transitionHandlers.Push(h)
		return nil
	})
	if err != nil {
		return err
	}

	return <-ready
}

func (p *process) Stop() os.Error {
	err := p.do(func() os.Error { return p.stopAsync() })
	if err != nil {
		return err
	}

	return p.WaitStop()
}

func (p *process) Detach() os.Error {
	if err := p.Stop(); err != nil {
		return err
	}

	err := p.do(func() os.Error {
		if err := p.uninstallBreakpoints(); err != nil {
			return err
		}

		for pid, t := range p.threads {
			if t.state.isStopped() {
				// We can't detach from zombies.
				if err := t.ptraceDetach(); err != nil {
					return err
				}
			}
			t.setState(detached)
			p.threads[pid] = nil, false
		}
		return nil
	})
	// TODO(austin) Wait for monitor thread to exit?
	return err
}

// newThread creates a new thread object and waits for its initial
// signal.  If cloned is true, this thread was cloned from a thread we
// are already attached to.
//
// Must be run from the monitor thread.
func (p *process) newThread(tid int, signal int, cloned bool) (*thread, os.Error) {
	t := &thread{tid: tid, proc: p, state: stopped}

	// Get the signal from the thread
	// TODO(austin) Thread might already be stopped if we're attaching.
	w, err := os.Wait(tid, syscall.WALL)
	if err != nil {
		return nil, err
	}
	if w.Pid != tid || w.StopSignal() != signal {
		return nil, &newThreadError{w, tid, signal}
	}

	if !cloned {
		err = t.ptraceSetOptions(syscall.PTRACE_O_TRACECLONE | syscall.PTRACE_O_TRACEEXIT)
		if err != nil {
			return nil, err
		}
	}

	p.threads[tid] = t

	return t, nil
}

// attachThread attaches a running thread to the process.
//
// Must NOT be run from the monitor thread.
func (p *process) attachThread(tid int) (*thread, os.Error) {
	p.logTrace("attaching to thread %d", tid)
	var thr *thread
	err := p.do(func() os.Error {
		errno := syscall.PtraceAttach(tid)
		if errno != 0 {
			return os.NewSyscallError("ptrace(ATTACH)", errno)
		}

		var err os.Error
		thr, err = p.newThread(tid, syscall.SIGSTOP, false)
		return err
	})
	return thr, err
}

// attachAllThreads attaches to all threads in a process.
func (p *process) attachAllThreads() os.Error {
	taskPath := "/proc/" + strconv.Itoa(p.pid) + "/task"
	taskDir, err := os.Open(taskPath, os.O_RDONLY, 0)
	if err != nil {
		return err
	}
	defer taskDir.Close()

	// We stop threads as we attach to them; however, because new
	// threads can appear while we're looping over all of them, we
	// have to repeatly scan until we know we're attached to all
	// of them.
	for again := true; again; {
		again = false

		tids, err := taskDir.Readdirnames(-1)
		if err != nil {
			return err
		}

		for _, tidStr := range tids {
			tid, err := strconv.Atoi(tidStr)
			if err != nil {
				return err
			}
			if _, ok := p.threads[tid]; ok {
				continue
			}

			_, err = p.attachThread(tid)
			if err != nil {
				// There could have been a race, or
				// this process could be a zobmie.
				statFile, err2 := ioutil.ReadFile(taskPath + "/" + tidStr + "/stat")
				if err2 != nil {
					switch err2 := err2.(type) {
					case *os.PathError:
						if err2.Error == os.ENOENT {
							// Raced with thread exit
							p.logTrace("raced with thread %d exit", tid)
							continue
						}
					}
					// Return the original error
					return err
				}

				statParts := strings.Split(string(statFile), " ", 4)
				if len(statParts) > 2 && statParts[2] == "Z" {
					// tid is a zombie
					p.logTrace("thread %d is a zombie", tid)
					continue
				}

				// Return the original error
				return err
			}
			again = true
		}
	}

	return nil
}

// newProcess creates a new process object and starts its monitor thread.
func newProcess(pid int) *process {
	p := &process{
		pid:         pid,
		threads:     make(map[int]*thread),
		breakpoints: make(map[uintptr]*breakpoint),
		ready:       make(chan bool, 1),
		debugEvents: make(chan *debugEvent),
		debugReqs:   make(chan *debugReq),
		stopReq:     make(chan os.Error),
	}

	go p.monitor()

	return p
}

// Attach attaches to process pid and stops all of its threads.
func Attach(pid int) (Process, os.Error) {
	p := newProcess(pid)

	// Attach to all threads
	err := p.attachAllThreads()
	if err != nil {
		p.Detach()
		// TODO(austin) Detach stopped the monitor already
		//p.stopMonitor(err);
		return nil, err
	}

	return p, nil
}

// ForkExec forks the current process and execs argv0, stopping the
// new process after the exec syscall.  See os.ForkExec for additional
// details.
func ForkExec(argv0 string, argv []string, envv []string, dir string, fd []*os.File) (Process, os.Error) {
	p := newProcess(-1)

	// Create array of integer (system) fds.
	intfd := make([]int, len(fd))
	for i, f := range fd {
		if f == nil {
			intfd[i] = -1
		} else {
			intfd[i] = f.Fd()
		}
	}

	// Fork from the monitor thread so we get the right tracer pid.
	err := p.do(func() os.Error {
		pid, errno := syscall.PtraceForkExec(argv0, argv, envv, dir, intfd)
		if errno != 0 {
			return &os.PathError{"fork/exec", argv0, os.Errno(errno)}
		}
		p.pid = pid

		// The process will raise SIGTRAP when it reaches execve.
		_, err := p.newThread(pid, syscall.SIGTRAP, false)
		return err
	})
	if err != nil {
		p.stopMonitor(err)
		return nil, err
	}

	return p, nil
}
