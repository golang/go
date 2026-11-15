// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package runtime

import "unsafe"

func LogosEnterFastCallForTest(arg unsafe.Pointer) (slots [4]uintptr, flag bool) {
	mp := getg().m
	logosRecordFastCall(mp, arg)
	return mp.fastCgoFrame, mp.inFastCall
}

func LogosExitFastCallForTest() (slots [4]uintptr, flag bool) {
	mp := getg().m
	logosClearFastCall(mp)
	return mp.fastCgoFrame, mp.inFastCall
}
func LogosFastCallStateForTest() (slots [4]uintptr, flag bool) {
	mp := getg().m
	return mp.fastCgoFrame, mp.inFastCall
}

func LogosDirtyFastCallSlotsForTest(slots [4]uintptr, flag bool) {
	mp := getg().m
	mp.fastCgoFrame = slots
	mp.inFastCall = flag
}

func LogosFastCallPassedMCheckForTest(arg unsafe.Pointer) (touchedCurrent, recordOK, clearOK bool) {
	cur := getg().m
	savedSlots := cur.fastCgoFrame
	savedFlag := cur.inFastCall
	fake := new(m)
	logosRecordFastCall(fake, arg)
	recordOK = fake.inFastCall && fake.fastCgoFrame[0] == uintptr(arg)
	touchedCurrent = cur.inFastCall != savedFlag || cur.fastCgoFrame != savedSlots
	logosClearFastCall(fake)
	clearOK = !fake.inFastCall && fake.fastCgoFrame == [4]uintptr{}
	cur.fastCgoFrame = savedSlots
	cur.inFastCall = savedFlag
	return touchedCurrent, recordOK, clearOK
}

func LogosGrowsliceBufForTest(elem any, buf unsafe.Pointer, bufLen, newLen, num int) (array unsafe.Pointer, length, capacity int) {
	s := growsliceBuf(buf, newLen, bufLen, num, efaceOf(&elem)._type, buf, bufLen)
	return s.array, s.len, s.cap
}

func SetLogosWaitAfterRegisterHookForTest(fn func()) {
	logosWaitAfterRegisterHook = fn
}
