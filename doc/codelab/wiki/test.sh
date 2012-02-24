#!/usr/bin/env bash

set -e
wiki_pid=
cleanup() {
	kill $wiki_pid
	rm -f test_*.out Test.txt final-test.bin final-test.go
}
trap cleanup 0 INT

make get.bin
addr=$(./get.bin -addr)
sed s/:8080/$addr/ < final.go > final-test.go
make final-test.bin
(./final-test.bin) &
wiki_pid=$!

sleep 1

./get.bin http://$addr/edit/Test > test_edit.out
diff -u test_edit.out test_edit.good
./get.bin -post=body=some%20content http://$addr/save/Test
diff -u Test.txt test_Test.txt.good
./get.bin http://$addr/view/Test > test_view.out
diff -u test_view.out test_view.good

echo PASS
