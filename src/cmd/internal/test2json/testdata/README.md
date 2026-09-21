# test2json test artifacts

This directory contains test artifacts for `TestGolden` in
[test2json_test.go](../test2json_test.go). For each set of `<test>.*` files:

- If `<test>.src` is present, TestGolden executes it as a script test and verifies
  that the output matches `<test>.test`. This verifies that the testing package
  produces the output expected by test2json.
- TestGolden reads `<test>.test` and processes it with a `Converter`, verifying
  that the output matches `<test>.json`.This verifies that test2json produces
  the expected output events.