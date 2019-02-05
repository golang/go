#!/bin/bash -x

# This is a script that generates the test MacOS executables in this directory.
# It should be needed very rarely to run this script. It is mostly provided
# as a future reference on how the original binary set was created.

set -o errexit

cat <<EOF >/tmp/hello.cc
#include <stdio.h>

int main() {
  printf("Hello, world!\n");
  return 0;
}
EOF

cat <<EOF >/tmp/lib.c
int foo() {
  return 1;
}

int bar() {
  return 2;
}
EOF

cd $(dirname $0)
rm -rf exe_mac_64* lib_mac_64*
clang -g -o exe_mac_64 /tmp/hello.c
clang -g -o lib_mac_64 -dynamiclib /tmp/lib.c
