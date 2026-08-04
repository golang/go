#include <dlfcn.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef int (*go_add_fn)(int, int);
typedef char* (*go_getenv_fn)(const char*);
typedef char* (*go_runtime_info_fn)(void);

static int test_basic_dlopen(const char* lib) {
	void* handle = dlopen(lib, RTLD_NOW | RTLD_GLOBAL);
	if (!handle) {
		fprintf(stderr, "FAIL basic_dlopen: %s\n", dlerror());
		return 1;
	}

	go_add_fn add = (go_add_fn)dlsym(handle, "GoAdd");
	if (!add) {
		fprintf(stderr, "FAIL basic_dlopen: missing GoAdd: %s\n", dlerror());
		dlclose(handle);
		return 1;
	}

	int result = add(40, 2);
	if (result != 42) {
		fprintf(stderr, "FAIL basic_dlopen: GoAdd(40,2)=%d, want 42\n", result);
		dlclose(handle);
		return 1;
	}

	dlclose(handle);
	printf("PASS basic_dlopen\n");
	return 0;
}

static int test_environment(const char* lib) {
	// Verify Go can read environment variables that exist from process start.
	// We check PATH since it's universally present. This exercises the
	// goenvs code path which walks argv to find envp (on glibc) or
	// reads /proc/self/environ (after our fix, in library mode).
	void* handle = dlopen(lib, RTLD_NOW | RTLD_GLOBAL);
	if (!handle) {
		fprintf(stderr, "FAIL environment: %s\n", dlerror());
		return 1;
	}

	go_getenv_fn getenv_fn = (go_getenv_fn)dlsym(handle, "GoGetenv");
	if (!getenv_fn) {
		fprintf(stderr, "FAIL environment: missing GoGetenv: %s\n", dlerror());
		dlclose(handle);
		return 1;
	}

	char* val = getenv_fn("PATH");
	if (!val || strlen(val) == 0) {
		fprintf(stderr, "FAIL environment: GoGetenv(PATH)=%s, want non-empty\n",
			val ? val : "(null)");
		free(val);
		dlclose(handle);
		return 1;
	}

	printf("PASS environment (PATH=%s)\n", val);
	free(val);
	dlclose(handle);
	return 0;
}

static int test_runtime_info(const char* lib) {
	void* handle = dlopen(lib, RTLD_NOW | RTLD_GLOBAL);
	if (!handle) {
		fprintf(stderr, "FAIL runtime_info: %s\n", dlerror());
		return 1;
	}

	go_runtime_info_fn info = (go_runtime_info_fn)dlsym(handle, "GoRuntimeInfo");
	if (!info) {
		fprintf(stderr, "FAIL runtime_info: missing GoRuntimeInfo: %s\n", dlerror());
		dlclose(handle);
		return 1;
	}

	char* s = info();
	if (!s || strlen(s) == 0) {
		fprintf(stderr, "FAIL runtime_info: empty result\n");
		free(s);
		dlclose(handle);
		return 1;
	}

	printf("PASS runtime_info (%s)\n", s);
	free(s);
	dlclose(handle);
	return 0;
}

int main(int argc, char** argv) {
	if (argc < 2) {
		fprintf(stderr, "usage: %s <path-to-libtest.so>\n", argv[0]);
		return 1;
	}

	const char* lib = argv[1];
	int failures = 0;

	failures += test_basic_dlopen(lib);
	failures += test_environment(lib);
	failures += test_runtime_info(lib);

	if (failures == 0) {
		printf("ALL PASS\n");
	} else {
		fprintf(stderr, "FAILURES: %d\n", failures);
	}
	return failures;
}
