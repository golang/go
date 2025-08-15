// Copyright 2025 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// This program loads a Go c-shared library using dlopen()
// to test non-glibc dlopen() behavior.

int main(int argc, char **argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <shared_library.so>\n", argv[0]);
        return 1;
    }

    // Set an environment variable before loading the library
    setenv("MUSL_TEST_VAR", "from_c", 1);

    // Try to load the shared library
    void *handle = dlopen(argv[1], RTLD_NOW);
    if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 1;
    }

    // Test 1: Call TestMuslInit to see if initialization succeeded
    void (*test_init)(void) = dlsym(handle, "TestMuslInit");
    if (!test_init) {
        fprintf(stderr, "dlsym TestMuslInit failed: %s\n", dlerror());
        dlclose(handle);
        return 1;
    }
    test_init();

    // Test 2: Check environment variable synchronization
    char* (*test_env)(void) = dlsym(handle, "TestEnvSync");
    if (!test_env) {
        fprintf(stderr, "dlsym TestEnvSync failed: %s\n", dlerror());
        dlclose(handle);
        return 1;
    }
    char* env_val = test_env();
    if (env_val && strcmp(env_val, "from_c") == 0) {
        printf("ENV_SYNC_SUCCESS\n");
    } else {
        printf("ENV_SYNC_FAIL: got '%s', expected 'from_c'\n", env_val ? env_val : "(null)");
    }

    // Test 3: Check if argc is accessible
    int (*test_argc)(void) = dlsym(handle, "TestArgc");
    if (test_argc) {
        int go_argc = test_argc();
        printf("ARGC_TEST: C=%d Go=%d\n", argc, go_argc);
    }

    // Test 4: Check if argv is accessible
    char* (*test_argv)(int) = dlsym(handle, "TestArgv");
    if (test_argv) {
        char* arg0 = test_argv(0);
        if (arg0 && *arg0) {
            printf("ARGV_TEST_SUCCESS: argv[0]=%s\n", arg0);
        } else {
            printf("ARGV_TEST_FAIL: argv[0] not accessible\n");
        }
    }

    dlclose(handle);
    return 0;
}
