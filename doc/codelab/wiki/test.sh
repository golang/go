#!/bin/bash

wiki_pid=

cleanup() {
	kill $wiki_pid
	rm -f test_*.out Test.txt final-test.bin final-test.go
	exit ${1:-1}
}
trap cleanup INT

port=$(./get.bin -port)
sed s/8080/$port/ < final.go > final-test.go
gomake final-test.bin || cleanup 1
./final-test.bin &
wiki_pid=$!

sleep 1

./get.bin http://127.0.0.1:$port/edit/Test > test_edit.out
diff -u test_edit.out test_edit.good || cleanup 1
./get.bin -post=body=some%20content http://127.0.0.1:$port/save/Test
diff -u Test.txt test_Test.txt.good || cleanup 1
./get.bin http://127.0.0.1:$port/view/Test > test_view.out
diff -u test_view.out test_view.good || cleanup 1

echo "Passed"
cleanup 0

