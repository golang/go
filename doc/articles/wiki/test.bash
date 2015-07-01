#!/usr/bin/env bash
# Copyright 2010 The Go Authors. All rights reserved.
# Use of this source code is governed by a BSD-style
# license that can be found in the LICENSE file.

set -e

if ! which patch > /dev/null; then
	echo "Skipping test; patch command not found."
	exit 0
fi

wiki_pid=
cleanup() {
	kill $wiki_pid
	rm -f test_*.out Test.txt final-test.go final-test.bin final-test-port.txt a.out get.bin
}
trap cleanup 0 INT

rm -f get.bin final-test.bin a.out

# If called with -all, check that all code snippets compile.
if [ "$1" == "-all" ]; then
	for fn in *.go; do
		go build -o a.out $fn
	done
fi

go build -o get.bin get.go
cp final.go final-test.go
patch final-test.go final-test.patch > /dev/null
go build -o final-test.bin final-test.go
./final-test.bin &
wiki_pid=$!

l=0
while [ ! -f ./final-test-port.txt ]
do
	l=$(($l+1))
	if [ "$l" -gt 5 ]
	then
		echo "port not available within 5 seconds"
		exit 1
		break
	fi
	sleep 1
done

addr=$(cat final-test-port.txt)
./get.bin http://$addr/edit/Test > test_edit.out
diff -u test_edit.out test_edit.good
./get.bin -post=body=some%20content http://$addr/save/Test > test_save.out
diff -u test_save.out test_view.good # should be the same as viewing
diff -u Test.txt test_Test.txt.good
./get.bin http://$addr/view/Test > test_view.out
diff -u test_view.out test_view.good

echo PASS
