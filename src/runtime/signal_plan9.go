// Copyright 2011 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

type sigTabT struct {
	flags int
	name  []byte
}

// Incoming notes are compared against this table using strncmp, so the
// order matters: longer patterns must appear before their prefixes.
// There are _SIG constants in os2_plan9.go for the table index of some
// of these.
//
// If you add entries to this table, you must respect the prefix ordering
// and also update the constant values is os2_plan9.go.
var sigtable = [...]sigTabT{
	// Traps that we cannot be recovered.
	{_SigThrow, []byte("sys: trap: debug exception")},
	{_SigThrow, []byte("sys: trap: invalid opcode")},

	// We can recover from some memory errors in runtime·sigpanic.
	{_SigPanic, []byte("sys: trap: fault read addr")},  // SIGRFAULT
	{_SigPanic, []byte("sys: trap: fault write addr")}, // SIGWFAULT

	// We can also recover from math errors.
	{_SigPanic, []byte("sys: trap: divide error")}, // SIGINTDIV
	{_SigPanic, []byte("sys: fp:")},                // SIGFLOAT

	// All other traps are normally handled as if they were marked SigThrow.
	// We mark them SigPanic here so that debug.SetPanicOnFault will work.
	{_SigPanic, []byte("sys: trap:")}, // SIGTRAP

	// Writes to a closed pipe can be handled if desired, otherwise they're ignored.
	{_SigNotify, []byte("sys: write on closed pipe")},

	// Other system notes are more serious and cannot be recovered.
	{_SigThrow, []byte("sys:")},

	// Issued to all other procs when calling runtime·exit.
	{_SigGoExit, []byte("go: exit ")},

	// Kill is sent by external programs to cause an exit.
	{_SigKill, []byte("kill")},

	// Interrupts can be handled if desired, otherwise they cause an exit.
	{_SigNotify + _SigKill, []byte("interrupt")},
	{_SigNotify + _SigKill, []byte("hangup")},

	// Alarms can be handled if desired, otherwise they're ignored.
	{_SigNotify, []byte("alarm")},
}
