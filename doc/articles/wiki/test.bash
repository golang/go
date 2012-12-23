#!/usr/bin/env bash
# Copyright 2010 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e
wiki_pid=
cleanup() {
	kill $wiki_pid
	rm -f test_*.out Test.txt final-test.bin final-test.go
}
trap cleanup 0 INT

go build -o get.bin get.go
addr=$(./get.bin -addr)
sed s/:8080/$addr/ < final.go > final-test.go
go build -o final-test.bin final-test.go
(./final-test.bin) &
wiki_pid=$!

./get.bin --wait_for_port=5s http://$addr/edit/Test > test_edit.out
diff -u test_edit.out test_edit.good
./get.bin -post=body=some%20content http://$addr/save/Test > test_save.out
diff -u test_save.out test_view.good # should be the same as viewing
diff -u Test.txt test_Test.txt.good
./get.bin http://$addr/view/Test > test_view.out
diff -u test_view.out test_view.good

echo PASS
