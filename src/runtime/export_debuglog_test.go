// Copyright 2019 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

// Export debuglog guts for testing.

package runtime

const DlogEnabled = dlogEnabled

const DebugLogBytes = debugLogBytes

const DebugLogStringLimit = debugLogStringLimit

type Dlogger = dloggerImpl

func Dlog() *Dlogger {
	return dlogImpl()
}

func (l *dloggerImpl) End()                      { l.end() }
func (l *dloggerImpl) B(x bool) *dloggerImpl     { return l.b(x) }
func (l *dloggerImpl) I(x int) *dloggerImpl      { return l.i(x) }
func (l *dloggerImpl) I16(x int16) *dloggerImpl  { return l.i16(x) }
func (l *dloggerImpl) U64(x uint64) *dloggerImpl { return l.u64(x) }
func (l *dloggerImpl) Hex(x uint64) *dloggerImpl { return l.hex(x) }
func (l *dloggerImpl) P(x any) *dloggerImpl      { return l.p(x) }
func (l *dloggerImpl) S(x string) *dloggerImpl   { return l.s(x) }
func (l *dloggerImpl) PC(x uintptr) *dloggerImpl { return l.pc(x) }

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
