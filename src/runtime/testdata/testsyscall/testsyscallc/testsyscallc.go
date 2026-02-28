// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package testsyscallc

/*
int syscall_check0(void) {
    return 1;
}

int syscall_check1(int a1) {
    return a1 == 1;
}

int syscall_check2(int a1, int a2) {
    return a1 == 1 && a2 == 2;
}

int syscall_check3(int a1, int a2, int a3) {
    return a1 == 1 && a2 == 2 && a3 == 3;
}

int syscall_check4(int a1, int a2, int a3, int a4) {
    return a1 == 1 && a2 == 2 && a3 == 3 && a4 == 4;
}

int syscall_check5(int a1, int a2, int a3, int a4, int a5) {
    return a1 == 1 && a2 == 2 && a3 == 3 && a4 == 4 && a5 == 5;
}

int syscall_check6(int a1, int a2, int a3, int a4, int a5, int a6) {
    return a1 == 1 && a2 == 2 && a3 == 3 && a4 == 4 && a5 == 5 && a6 == 6;
}

int syscall_check7(int a1, int a2, int a3, int a4, int a5, int a6, int a7) {
    return a1 == 1 && a2 == 2 && a3 == 3 && a4 == 4 && a5 == 5 && a6 == 6 && a7 == 7;
}

int syscall_check8(int a1, int a2, int a3, int a4, int a5, int a6, int a7, int a8) {
    return a1 == 1 && a2 == 2 && a3 == 3 && a4 == 4 && a5 == 5 && a6 == 6 && a7 == 7 && a8 == 8;
}

int syscall_check9(int a1, int a2, int a3, int a4, int a5, int a6, int a7, int a8, int a9) {
    return a1 == 1 && a2 == 2 && a3 == 3 && a4 == 4 && a5 == 5 && a6 == 6 && a7 == 7 && a8 == 8 && a9 == 9;
}
*/
import "C"
