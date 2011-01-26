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

curl -s -o test_edit.out http://localhost:8080/edit/Test 
diff -u test_edit.out test_edit.good || cleanup 1
curl -s -o /dev/null -d body=some%20content http://localhost:8080/save/Test
diff -u Test.txt test_Test.txt.good || cleanup 1
curl -s -o test_view.out http://localhost:8080/view/Test
diff -u test_view.out test_view.good || cleanup 1

echo "Passed"
cleanup 0

