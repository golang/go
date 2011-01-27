#!/bin/bash

./final.bin &
wiki_pid=$!

cleanup() {
	kill $wiki_pid
	rm -f test_*.out Test.txt
	exit ${1:-1}
}
trap cleanup INT

sleep 1

./get.bin http://localhost:8080/edit/Test > test_edit.out
diff -u test_edit.out test_edit.good || cleanup 1
./get.bin -post=body=some%20content http://localhost:8080/save/Test
diff -u Test.txt test_Test.txt.good || cleanup 1
./get.bin http://localhost:8080/view/Test > test_view.out
diff -u test_view.out test_view.good || cleanup 1

echo "Passed"
cleanup 0

