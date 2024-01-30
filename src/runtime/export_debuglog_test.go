// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Export debuglog guts for testing.

package runtime

const DlogEnabled = dlogEnabled

const DebugLogBytes = debugLogBytes

const DebugLogStringLimit = debugLogStringLimit

var Dlog = dlog

func (l *dlogger) End()                  { l.end() }
func (l *dlogger) B(x bool) *dlogger     { return l.b(x) }
func (l *dlogger) I(x int) *dlogger      { return l.i(x) }
func (l *dlogger) I16(x int16) *dlogger  { return l.i16(x) }
func (l *dlogger) U64(x uint64) *dlogger { return l.u64(x) }
func (l *dlogger) Hex(x uint64) *dlogger { return l.hex(x) }
func (l *dlogger) P(x any) *dlogger      { return l.p(x) }
func (l *dlogger) S(x string) *dlogger   { return l.s(x) }
func (l *dlogger) PC(x uintptr) *dlogger { return l.pc(x) }

func DumpDebugLog() string {
	gp := getg()
	gp.writebuf = make([]byte, 0, 1<<20)
	printDebugLog()
	buf := gp.writebuf
	gp.writebuf = nil

	return string(buf)
}

func ResetDebugLog() {
	stw := stopTheWorld(stwForTestResetDebugLog)
	for l := allDloggers; l != nil; l = l.allLink {
		l.w.write = 0
		l.w.tick, l.w.nano = 0, 0
		l.w.r.begin, l.w.r.end = 0, 0
		l.w.r.tick, l.w.r.nano = 0, 0
	}
	startTheWorld(stw)
}
