// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

//go:build goexperiment.jit

package runtime

func verifyUserFrameTree(n *userFrameNode) (height, count int, low, high uintptr, ok bool) {
	if n == nil {
		return 0, 0, 0, 0, true
	}
	lh, lc, ll, lr, lok := verifyUserFrameTree(n.left)
	rh, rc, rl, rr, rok := verifyUserFrameTree(n.right)
	ok = lok && rok && lh-rh <= 1 && rh-lh <= 1 && n.height == 1+max(lh, rh)
	ok = ok && (lc == 0 || lr <= n.region.start) && (rc == 0 || rl >= n.region.end)
	low, high = n.region.start, n.region.end
	if lc != 0 {
		low = ll
	}
	if rc != 0 {
		high = rr
	}
	return n.height, lc + rc + 1, low, high, ok
}

func UserFramePersistentTreeForTest() bool {
	const count = 4096
	var root, retained *userFrameNode
	for i := 0; i < count; i++ {
		index := (i * 4051) % count
		r := &userFrameRegion{start: uintptr(index * 16), end: uintptr(index*16 + 8)}
		root = userFrameInsert(root, r)
		if i == 1023 {
			retained = root
		}
	}
	GC()
	_, total, _, _, ok := verifyUserFrameTree(root)
	if !ok || total != count {
		return false
	}
	for i := 0; i < count; i++ {
		index := (i * 4051) % count
		root = userFrameRemove(root, uintptr(index*16))
		if i%37 == 0 {
			_, total, _, _, ok = verifyUserFrameTree(root)
			if !ok || total != count-i-1 {
				return false
			}
		}
	}
	GC()
	_, total, _, _, ok = verifyUserFrameTree(retained)
	return root == nil && ok && total == 1024
}

func UserFrameStableStackMapBlocksForTest() bool {
	table := new(userFrameStackMapTable)
	var first *userFrameStackMap
	var retained *userFrameStackMapIndex
	for i := 0; i < 16385; i++ {
		mask := []byte{byte(i), 1}
		input := []userFrameStackMapInput{{pcOffset: uintptr(i * 8), frameWords: 16, pointerMask: mask}}
		appendUserFrameStackMaps(table, 0, 1<<20, userFrameUnwindStop, input)
		mask[0] = 0
		if i == 0 {
			first = userFrameMapAt(table.index, 0)
			retained = table.index
		}
		if i == 64 || i == 1024 || i == 16384 {
			GC()
			if userFrameMapAt(table.index, 0) != first || userFrameMapAt(retained, 0) != first {
				return false
			}
		}
	}
	for i := 0; i < 16385; i++ {
		m := userFrameMapAt(table.index, uintptr(i))
		if m.pcOffset != uintptr(i*8) || m.frameWords != 16 || *m.pointerMask != byte(i) {
			return false
		}
	}
	return true
}
